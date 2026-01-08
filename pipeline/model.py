import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class HistologyProjector(nn.Module):
    def __init__(self, visual_dim=1024, llm_dim=3584):
        super().__init__()
        # Linear layer aligns visual feature space to LLM space
        self.proj = nn.Linear(visual_dim, llm_dim)
        
    def forward(self, x):
        return self.proj(x)

class SlideQwenModel(nn.Module):
    def __init__(self, base_model_path, slide_dim=1024):
        super().__init__()
        # Load the base architecture
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        # Initialize the projection layer
        self.projector = HistologyProjector(visual_dim=slide_dim, llm_dim=self.llm.config.hidden_size)

    def forward(self, knowledge_embeds, slide_features, input_ids, labels=None):
        """
        Forward path for training or inference.
        Sequence: [Knowledge] + [Projected Slides] + [User Prompt]
        """
        # 1. Project Slide Features
        projected_slides = self.projector(slide_features)
        
        # 2. Get Text Embeddings from Qwen
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Concatenate all modalities into one sequence
        # Shape: (Batch, Total_Seq, 3584)
        inputs_embeds = torch.cat([knowledge_embeds, projected_slides, text_embeds], dim=1)
        
        # 4. Pass through LLM
        return self.llm(inputs_embeds=inputs_embeds, labels=labels)

    @torch.no_grad()
    def generate_report(self, knowledge_embeds, slide_features, prompt_text, max_tokens=512):
        # Prepare text prompt without chat roles
        input_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.llm.device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Project slide features
        projected_slides = self.projector(slide_features)
        
        # Merge embeddings for decoding
        inputs_embeds = torch.cat([knowledge_embeds, projected_slides, text_embeds], dim=1)
        
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)