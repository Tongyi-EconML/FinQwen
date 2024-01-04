import re


class PromptGenerator:

    def __init__(self,
                 plan_template: str = '',
                 task_template: str = '',
                 task_instruction_template: str = '',
                 user_template: str = '',
                 current_task_template: str = '',
                 sep='\n\n',
                 prompt_max_length: int = 10000):
        """
        prompt genertor
        Args:
            system_template (str, optional): System template, normally the role of LLM.
            instruction_template (str, optional): Indicate the instruction for LLM.
            user_template (str, optional): Prefix before user input. Defaults to ''.
            exec_template (str, optional): A wrapper str for exec result.
            assistant_template (str, optional): Prefix before assistant response.
            Some LLM need to manully concat this prefix before generation.
            prompt_max_length (int, optional): max length of prompt. Defaults to 2799.

        """

        self.plan_template = plan_template
        self.task_template = task_template
        self.task_instruction_template = task_instruction_template
        self.user_template = user_template
        self.current_task_template = current_task_template
        self.sep = sep

        self.prompt_max_length = prompt_max_length
        self.reset()

    def reset(self):
        self.prompt = ''

    def init_plan_prompt(self, user_question):
        """
        in this function, the prompt will be initialized.
        """
        self.system_prompt = self.plan_template
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        self.current_task_prompt = None
        self.task_result_prompt = None


    def init_task_prompt(self,user_question, tool_list):
        self.system_prompt = self.task_template + self.task_instruction_template.replace("{tool_list}",self.get_tool_str(tool_list))
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        self.current_task_prompt = self.current_task_template
        self.task_result_prompt = None

    def generate(self):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        pass

    def get_tool_str(self, tool_list):
        """generate tool list string

        Args:
            tool_list (List[str]): list of tools

        """
        tool_str = self.sep.join(
            [f'{i+1}. {t}' for i, t in enumerate(tool_list)])
        return tool_str

    def get_history_str(self):
        """generate history string

        """
        history_str = ''
        for i in range(len(self.history)):
            history_item = self.history[len(self.history) - i - 1]
            text = history_item['content']
            if len(history_str) + len(text) + len(
                    self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str
