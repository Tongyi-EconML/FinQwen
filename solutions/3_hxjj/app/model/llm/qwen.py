from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
from .base import LLM

class Qwen(LLM):
    model_type = 'qwen'
    def __init__(self,cfg):
        model_path = cfg.get("model_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True)
        self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
        # 加载原模型
        if cfg.get("lora_path") is not None:
            self.model = PeftModel.from_pretrained(self.model, cfg["lora_path"], adapter_name='agent_lora')
        if cfg.get("lora_2_path") is not None:
            self.model.load_adapter(cfg["lora_2_path"], adapter_name='sql_lora')
        
    
    def original_generate(self, prompt, history=None):
        with self.model.disable_adapter():
            response, _ = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
            return response

    def generate(self,prompt,history=None):
        self.model.set_adapter('agent_lora')
        response, _ = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
        return response
        
    def sql_generate(self,prompt,history=None):
        self.model.set_adapter('sql_lora')
        response, _ = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
        return response