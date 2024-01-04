from .llm import *
from config.model_config import LLM_MODEL_DICT

def get_llm_cls(llm_type):
    if llm_type == 'qwen':
        return Qwen
    # elif llm_type == 'baichuan':
    #     return Baichuan
    # elif llm_type == 'chatglm':
    #     return ChatGLM
    # elif llm_type == 'test':
    #     return TestLM
    else:
        raise ValueError(f'Invalid llm_type {llm_type}')


class LLMFactory:
    @staticmethod
    def build_llm(model_name, additional_cfg):
        cfg = LLM_MODEL_DICT.get(model_name, {'type': 'test'})
        cfg.update(additional_cfg)
        llm_type = cfg.pop('type')
        llm_cls = get_llm_cls(llm_type)
        llm_cfg = cfg
        return llm_cls(cfg=llm_cfg)
    
