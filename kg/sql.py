import pymysql
import csv
import sys
from argparse import ArgumentParser
import pymysql.cursors

# Increase field size limit just in case (Windows compatible)
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument('--user', default='root', type=str)
    parser.add_argument('--password', default='1111', type=str)
    parser.add_argument('--database', required=False, type=str, default='umlsv1')
    
    args = parser.parse_args()

    print(f"Connecting to {args.database}...")
    # Use SSCursor (Server Side Cursor) to stream rows instead of loading all into RAM
    conn = pymysql.connect(
        host=args.host, user=args.user, password=args.password, database=args.database,
        cursorclass=pymysql.cursors.SSCursor 
    )
    
    valid_cuis = set()
    definitions = {}
    
    with conn.cursor() as cursor:
        # Get English definitions
        sql = "SELECT CUI, DEF FROM MRDEF"
        cursor.execute(sql)
        
        with open('definitions.csv', 'w', encoding='utf-8', newline='') as def_file:
            def_writer = csv.writer(def_file)
            def_writer.writerow(['cui:ID', 'definition'])
            
            while True:
                row = cursor.fetchone()
                if not row:
                    break
                cui, definition = row
                # Store only the first definition for each CUI
                if cui not in definitions and definition:
                    definitions[cui] = definition
                    def_writer.writerow([cui, definition])
    
    print(f"Loaded {len(definitions)} definitions.")


    
    # Header format for neo4j-admin import
    # :ID defines the unique ID for import
    # :LABEL defines the node label
    concept_cols = ['cui:ID', 'name', 'description', ':LABEL']
    
    with open('nodes.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(concept_cols)
        
        with conn.cursor() as cursor:
            # Select only English terms to keep graph clean but in newer subset ENG is only used                     
            # Only select important columns -> CUI (0), STR (14 for name) in mrconso
            # There is exactly one row in this file for each atom. So CUIs are repeated                            
            sql = "SELECT CUI, LAT, STR FROM MRCONSO WHERE LAT='ENG'"
            cursor.execute(sql)
            
            while True:
                row = cursor.fetchone()
                if not row:
                    break
                
                cui, lat, name = row
                
                # Only write the CUI if we haven't seen it yet
                if cui not in valid_cuis:
                    # Get definition if available, otherwise empty string
                    description = definitions.get(cui, '')
                    writer.writerow([cui, name, description, 'Concept'])
                    valid_cuis.add(cui)

    print(f"Saved {len(valid_cuis)} unique concepts.")

    # relationships in mrrel
    print('Processing Relationships (Edges)...')
    
    # Header format for neo4j-admin import
    # :START_ID refers to the CUI from nodes.csv
    # :END_ID refers to the CUI from nodes.csv
    # :TYPE is the relationship type (e.g., ISA, RO)
    rel_cols = [':START_ID', ':END_ID', ':TYPE', 'RELA']
    
    count = 0
    with open('relationships.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(rel_cols)
        
        with conn.cursor() as cursor:
            # fetch CUI1, CUI2, REL(main type), RELA(sub type)
            # filter CUI1 != CUI2 to avoid self-loops
            sql = "SELECT CUI1, CUI2, REL, RELA FROM MRREL WHERE CUI1 != CUI2"
            cursor.execute(sql)
            
            while True:
                row = cursor.fetchone()
                if not row:
                    break
                
                cui1, cui2, rel, rela = row
                
                # Ensure we only write relationships between nodes that actually exist
                if cui1 in valid_cuis and cui2 in valid_cuis:
                    # Use REL (PAR/CHD) as the main type, and RELA (isa/part_of) as a property
                    # If RELA is missing (None), simplify.
                    safe_rela = rela if rela else ""
                    writer.writerow([cui1, cui2, rel, safe_rela])
                    count += 1
                
                if count % 1000000 == 0 and count > 0:
                    print(f"  Processed {count} edges...")

    print(f"Finished. Saved {count} relationships.")
    conn.close()