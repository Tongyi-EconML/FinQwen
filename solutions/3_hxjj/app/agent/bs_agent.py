from copy import deepcopy
from typing import Dict, List, Optional, Union
import re

from model.llm import LLM
from .output_parser import BSOutputParser
from .output_wrapper import display
from prompt import BSPromptGenerator
# from ..retrieve import ToolRetrieval
from chains import DocumentRetrieveChain, DatabaseQueryChain, SqlGeneratorChain
# 构建Agent，需要传入llm，工具配置config以及工具检索

DEFAULT_TOOL_LIST = {"DocumentRetrieve": DocumentRetrieveChain(),
                     "SqlGenerator": SqlGeneratorChain(),
                        "DatabaseQuery": DatabaseQueryChain()}

class BSAgentExecutor:

    def __init__(self,
                 llm: LLM,
                 display: bool = False):
        """
        the core class of ms agent. It is responsible for the interaction between user, llm and tools,
        and return the execution result to user.

        Args:
            llm (LLM): llm model, can be load from local or a remote server.
            tool_cfg (Optional[Dict]): cfg of default tools
            additional_tool_list (Optional[Dict], optional): user-defined additional tool list. Defaults to {}.
            prompt_generator (Optional[PromptGenerator], optional): this module is responsible for generating prompt
            according to interaction result. Defaults to use MSPromptGenerator.
            output_parser (Optional[OutputParser], optional): this module is responsible for parsing output of llm
            to executable actions. Defaults to use MsOutputParser.
            tool_retrieval (Optional[Union[bool, ToolRetrieval]], optional): Retrieve related tools by input task,
            since most of tools may be uselees for LLM in specific task.
            If is bool type and it is True, will use default tool_retrieval. Defaults to True.
            knowledge_retrieval (Optional[KnowledgeRetrieval], optional): If user want to use extra knowledge,
            this component can be used to retrieve related knowledge. Defaults to None.
        """

        self.llm = llm
        self._init_tools()
        self.prompt_generator = BSPromptGenerator()
        self.output_parser = BSOutputParser()
        self.task_list = []
        self.task_no = None
        self.display = display
        self.agent_state = {}
        self.error_nums = 0

    def _init_tools(self):
        """init tool list of agent. We provide a default tool list, which is initialized by a cfg file.
        user can also provide user-defined tools by additional_tool_list.
        The key of additional_tool_list is tool name, and the value is corresponding object.

        Args:
            tool_cfg (Dict): default tool cfg.
            additional_tool_list (Dict, optional): user-defined tools. Defaults to {}.
        """
        self.available_tool_list = deepcopy(DEFAULT_TOOL_LIST)

    def run(self,
            user_input: str,
            print_info: bool = False) -> List[Dict]:
        """ use llm and tools to execute task given by user

        Args:
            task (str): concrete task
            remote (bool, optional): whether to execute tool in remote mode. Defaults to False.
            print_info (bool, optional): whether to print prompt info. Defaults to False.

        Returns:
            List[Dict]: execute result. One task may need to interact with llm multiple times,
            so a list of dict is returned. Each dict contains the result of one interaction.
        """

        # no task, first generate task list
        
        
        idx = 0
        final_res = []
        self.reset()
        self.agent_state["用户问题"] = user_input
        self.prompt_generator.init_plan_prompt(user_input)
        
        while True:
            idx += 1
            # generate prompt and call llm
            llm_result, exec_result = '', {}
            prompt = self.prompt_generator.generate(self.task_no)
            llm_result = self.llm.generate(prompt)
            if print_info:
                print(f'|prompt{idx}: {prompt}')
                print(f'|llm_result{idx}: {llm_result}')

            # parse and get tool name and arguments
            action, action_args = self.output_parser.parse_response(llm_result)
            if print_info:
                print(f'|action: {action}, action_args: {action_args}')
            final_res.append(llm_result)
            if action is None:
                # in chat mode, the final result of last instructions should be updated to prompt history
                pass
            elif action in self.available_tool_list:
                action_args = self.parse_action_args(action_args)
                tool = self.available_tool_list[action]
                try:
                    exec_result = tool(user_input,**action_args)
                    if print_info:
                        print(f'|exec_result: {exec_result}')
                    # parse exec result and store result to agent state
                    if exec_result.get("error") is not None:
                        final_res.append(exec_result)
                        if self.error_nums < 3:
                            self.error_nums += 1
                            self.prompt_generator.init_plan_prompt(user_input)
                            self.task_no = None
                            continue
                        return final_res
                    final_res.append(exec_result.get('result', ''))
                    self.parse_exec_result(exec_result)
                except Exception as e:
                    exec_result = f'Action call error: {action}: {action_args}. \n Error message: {e}'
                    final_res.append({'error': exec_result})
                    if self.error_nums < 3:
                        self.error_nums += 1
                        self.prompt_generator.init_plan_prompt(user_input)
                        self.task_no = None
                        continue
                    return final_res
            else:
                exec_result = f"Unknown action: '{action}'. "
                final_res.append({'error': exec_result})
                if self.error_nums < 3:
                    self.error_nums += 1
                    self.prompt_generator.init_plan_prompt(user_input)
                    self.task_no = None
                    continue
                return final_res

            # display result
            if self.display:
                display(llm_result, exec_result, idx)
            if self.task_no is None:
                self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                self.task_no = 0
                self.task_list = [task.strip() for task in re.split(r'\n[0-9].',llm_result)[1:]]
            else:
                self.task_no += 1
                if self.task_no >= len(self.task_list):
                    return final_res
            
            self.prompt_generator.update_task_prompt(self.task_list[self.task_no])


    def reset(self):
        """
        clear history and agent state
        """
        # self.prompt_generator.reset()
        self.task_no = None
        self.exec_result = ''
        self.agent_state = dict()
        self.error_nums = 0

    def parse_action_args(self, action_args):
        """
        replace action_args in str to Image/Video/Audio Wrapper, so that tool can handle them
        """
        parsed_action_args = {}
        for name, arg in action_args.items():
            try:
                true_arg = self.agent_state.get(arg, arg)
            except Exception:
                true_arg = arg
            parsed_action_args[name] = true_arg
        return parsed_action_args

    def parse_exec_result(self, exec_result):
        """
        update exec result to agent state.
        key is the str representation of the result.
        """
        self.agent_state[f"任务{self.task_no+1}的返回结果"] = exec_result["result"]
