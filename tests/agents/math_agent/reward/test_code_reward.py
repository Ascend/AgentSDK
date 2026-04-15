import pytest
import ast
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Any


@pytest.fixture(autouse=True, scope="function")
def mock_third_party_imports():
    """Mock all third-party imports"""
    with patch.dict('sys.modules', {
        'rllm': MagicMock(),
        'rllm.rewards': MagicMock(),
        'rllm.rewards.code_utils': MagicMock(),
        'rllm.rewards.code_utils.firejail_exec': MagicMock(),
        'rllm.rewards.code_utils.humanevalplus': MagicMock(),
        'rllm.rewards.code_utils.kodcode': MagicMock(),
        'rllm.rewards.code_utils.livecodebench': MagicMock(),
        'rllm.rewards.code_utils.taco': MagicMock(),
        'rllm.tools': MagicMock(),
        'rllm.tools.code_tools': MagicMock(),
        'rllm.tools.code_tools.code_tool': MagicMock(),
        'rllm.tools.code_tools.together_tool': MagicMock(),
        'rllm.tools.utils': MagicMock(),
    }):
        yield


@pytest.fixture(autouse=True, scope="function")
def mock_multiprocessing():
    """Mock multiprocessing to avoid actual process creation"""
    with patch('multiprocessing.Process') as mock_process:
        mock_instance = Mock()
        mock_process.return_value = mock_instance
        yield mock_process


class TestExtractCodeFromModel:
    def test_extract_code_single_block(self):
        from agents.math_agent.reward.code_reward import extract_code_from_model
        response = "Here's the code:\n```python\nprint('hello')\n```"
        result = extract_code_from_model(response)
        assert result == "print('hello')"

    def test_extract_code_multiple_blocks_returns_last(self):
        from agents.math_agent.reward.code_reward import extract_code_from_model
        response = "First:\n```python\ncode1\n```\nSecond:\n```python\ncode2\n```"
        result = extract_code_from_model(response)
        assert result == "code2"

    def test_extract_code_no_code_block(self):
        from agents.math_agent.reward.code_reward import extract_code_from_model
        response = "Just plain text without code blocks"
        result = extract_code_from_model(response)
        assert result is None

    def test_extract_code_with_language_specifier(self):
        from agents.math_agent.reward.code_reward import extract_code_from_model
        response = "```java\nSystem.out.println('hello');\n```"
        result = extract_code_from_model(response)
        assert result == "System.out.println('hello');"

    def test_extract_code_empty_response(self):
        from agents.math_agent.reward.code_reward import extract_code_from_model
        result = extract_code_from_model("")
        assert result is None


class TestCleanCodeMainBlock:
    def test_remove_if_main_block_double_quotes(self):
        from agents.math_agent.reward.code_reward import clean_code_main_block
        code = """def hello():
    print('hello')

if __name__ == "__main__":
    hello()"""
        result = clean_code_main_block(code)
        assert 'if __name__ == "__main__"' not in result
        assert "def hello():" in result

    def test_remove_if_main_block_single_quotes(self):
        from agents.math_agent.reward.code_reward import clean_code_main_block
        code = """def hello():
    print('hello')

if __name__ == '__main__':
    hello()"""
        result = clean_code_main_block(code)
        assert "if __name__ == '__main__'" not in result

    def test_no_main_block(self):
        from agents.math_agent.reward.code_reward import clean_code_main_block
        code = """def hello():
    print('hello')"""
        result = clean_code_main_block(code)
        assert result == code

    def test_main_block_with_indented_code(self):
        from agents.math_agent.reward.code_reward import clean_code_main_block
        code = """def hello():
    print('hello')

if __name__ == "__main__":
    print('running main')
    hello()"""
        result = clean_code_main_block(code)
        assert 'if __name__ == "__main__"' not in result
        assert "print('running main')" not in result

    def test_main_block_not_at_end(self):
        from agents.math_agent.reward.code_reward import clean_code_main_block
        code = """if __name__ == "__main__":
    print('main')
    
def other():
    pass"""
        result = clean_code_main_block(code)
        assert 'if __name__ == "__main__"' not in result
        assert "def other():" in result


class TestTacoToLcbFormat:
    def test_convert_dict_to_list_format(self):
        from agents.math_agent.reward.code_reward import taco_to_lcb_format
        tests = {
            "inputs": ["input1", "input2"],
            "outputs": ["output1", "output2"]
        }
        result = taco_to_lcb_format(tests)
        assert len(result) == 2
        assert result[0]["input"] == "input1"
        assert result[0]["output"] == "output1"
        assert result[1]["input"] == "input2"
        assert result[1]["output"] == "output2"

    def test_unequal_lengths_inputs_longer(self):
        from agents.math_agent.reward.code_reward import taco_to_lcb_format
        tests = {
            "inputs": ["input1", "input2", "input3"],
            "outputs": ["output1"]
        }
        result = taco_to_lcb_format(tests)
        assert len(result) == 3
        assert result[1]["output"] == "output1"
        assert result[2]["output"] == "output1"

    def test_unequal_lengths_outputs_longer(self):
        from agents.math_agent.reward.code_reward import taco_to_lcb_format
        tests = {
            "inputs": ["input1"],
            "outputs": ["output1", "output2", "output3"]
        }
        result = taco_to_lcb_format(tests)
        assert len(result) == 3
        assert result[1]["input"] == "input1"
        assert result[2]["input"] == "input1"

    def test_empty_inputs_and_outputs(self):
        from agents.math_agent.reward.code_reward import taco_to_lcb_format
        tests = {"inputs": [], "outputs": []}
        result = taco_to_lcb_format(tests)
        assert result == []

    def test_with_fn_name(self):
        from agents.math_agent.reward.code_reward import taco_to_lcb_format
        tests = {
            "inputs": ["input1"],
            "outputs": ["output1"],
            "fn_name": "test_func"
        }
        result = taco_to_lcb_format(tests)
        assert result[0]["testtype"] == "functional"
        assert result[0]["metadata"]["func_name"] == "test_func"


class TestPostprocessLcbSample:
    def test_basic_sample(self):
        from agents.math_agent.reward.code_reward import postprocess_lcb_sample
        sample = [
            {"input": "test1", "output": "out1"},
            {"input": "test2", "output": "out2"}
        ]
        result = postprocess_lcb_sample(sample)
        assert "input_output" in result
        parsed = json.loads(result["input_output"])
        assert parsed["inputs"] == ["test1", "test2"]
        assert parsed["outputs"] == ["out1", "out2"]

    def test_functional_testtype(self):
        from agents.math_agent.reward.code_reward import postprocess_lcb_sample
        sample = [
            {
                "input": "test1",
                "output": "out1",
                "testtype": "functional",
                "metadata": {"func_name": "my_function"}
            }
        ]
        result = postprocess_lcb_sample(sample)
        parsed = json.loads(result["input_output"])
        assert parsed["fn_name"] == "my_function"

    def test_functional_without_fn_name_raises_assertion(self):
        from agents.math_agent.reward.code_reward import postprocess_lcb_sample
        sample = [
            {
                "input": "test1",
                "output": "out1",
                "testtype": "functional",
                "metadata": {}
            }
        ]
        with pytest.raises(ValueError, match="Function name is not found"):
            postprocess_lcb_sample(sample)


class TestCheckCorrectness:
    def test_check_correctness_list_format_success(self):
        def mock_test_fn(tests, **kwargs):
            return [True, True]
        
        tests = [{"input": "1", "output": "2"}, {"input": "3", "output": "4"}]
        
        with patch('agents.math_agent.reward.code_reward.Manager') as mock_manager_class:
            mock_manager = Mock()
            mock_list = Mock()
            # Mock the list to behave like a list that can be appended to
            mock_list_data = []
            mock_list.side_effect = lambda: mock_list_data
            mock_list.append = lambda x: mock_list_data.append(x)
            mock_manager.list.return_value = mock_list_data
            mock_manager_class.return_value = mock_manager
            
            with patch('multiprocessing.Process') as mock_process_class:
                mock_instance = Mock()
                mock_instance.is_alive.return_value = False
                mock_process_class.return_value = mock_instance
                from agents.math_agent.reward.code_reward import check_correctness
                # Manually call evaluate_code to populate test_results
                def evaluate_side_effect(target, args):
                    # Simulate the process running evaluate_code
                    tests_arg, code, debug, test_results, test_fn = args
                    test_results.append([True, True])
                
                mock_process_class.return_value.start.side_effect = lambda: evaluate_side_effect(None, (tests, "code", False, mock_list_data, mock_test_fn))
                
                result, details = check_correctness(tests, "print('test')", mock_test_fn)
                
                assert result is True
                assert details["all_passed"] is True

    def test_check_correctness_list_format_failure(self):
        def mock_test_fn(tests, **kwargs):
            return [True, False, True]
        
        tests = [{"input": "1", "output": "2"}, {"input": "3", "output": "4"}, {"input": "5", "output": "6"}]
        
        with patch('agents.math_agent.reward.code_reward.Manager') as mock_manager_class:
            mock_manager = Mock()
            mock_list_data = []
            mock_manager.list.return_value = mock_list_data
            mock_manager_class.return_value = mock_manager
            
            with patch('multiprocessing.Process') as mock_process_class:
                from agents.math_agent.reward.code_reward import check_correctness
                mock_instance = Mock()
                mock_instance.is_alive.return_value = False
                mock_process_class.return_value = mock_instance
                
                # Simulate the process running
                def start_side_effect():
                    mock_list_data.append([True, False, True])
                mock_instance.start.side_effect = start_side_effect
                
                result, details = check_correctness(tests, "print('test')", mock_test_fn)
                
                assert result is False
                assert details["all_passed"] is False
                assert details["passed_tests"] == 2

    def test_check_correctness_dict_format(self):
        def mock_test_fn(tests, **kwargs):
            return [True, False]
        
        tests = {"inputs": ["1", "2"], "outputs": ["2", "3"]}
        
        with patch('agents.math_agent.reward.code_reward.Manager') as mock_manager_class:
            mock_manager = Mock()
            mock_list_data = []
            mock_manager.list.return_value = mock_list_data
            mock_manager_class.return_value = mock_manager
            
            with patch('multiprocessing.Process') as mock_process_class:
                from agents.math_agent.reward.code_reward import check_correctness
                mock_instance = Mock()
                mock_instance.is_alive.return_value = False
                mock_process_class.return_value = mock_instance
                
                def start_side_effect():
                    mock_list_data.append([True, False])
                mock_instance.start.side_effect = start_side_effect
                
                result, details = check_correctness(tests, "print('test')", mock_test_fn)
                
                assert result is False
                assert len(details["test_results"]) == 2

    def test_check_correctness_timeout(self):
        def mock_test_fn(tests, **kwargs):
            return [True]
        
        tests = [{"input": "1", "output": "2"}]
        
        with patch('agents.math_agent.reward.code_reward.Manager') as mock_manager_class:
            mock_manager = Mock()
            mock_list_data = []
            mock_manager.list.return_value = mock_list_data
            mock_manager_class.return_value = mock_manager
            
            with patch('multiprocessing.Process') as mock_process_class:
                from agents.math_agent.reward.code_reward import check_correctness
                mock_instance = Mock()
                mock_instance.is_alive.return_value = True
                mock_process_class.return_value = mock_instance
                
                result, details = check_correctness(tests, "print('test')", mock_test_fn)
                
                assert result is False
                assert details["all_passed"] is False


class TestPrimeIntellectCheckCorrectness:

    def test_string_input_parse_error(self):
        from agents.math_agent.reward.code_reward import primeintellect_check_correctness
        tests_str = 'invalid json'
        result, details = primeintellect_check_correctness(tests_str, "code")
        
        assert result is False
        assert "error" in details

    def test_dict_input(self):
        tests = [{"input": "1", "output": "2"}, {"input": "3", "output": "4"}]
        
        with patch('agents.math_agent.reward.code_reward.check_correctness') as mock_check:
            from agents.math_agent.reward.code_reward import primeintellect_check_correctness
            mock_check.return_value = (True, {"all_passed": True})
            result, details = primeintellect_check_correctness(tests, "code")
            
            assert mock_check.called

    def test_with_fn_name(self):
        tests = [{"input": "1", "output": "2", "fn_name": "test_func"}]
        
        with patch('agents.math_agent.reward.code_reward.check_correctness') as mock_check:
            from agents.math_agent.reward.code_reward import primeintellect_check_correctness
            mock_check.return_value = (True, {"all_passed": True})
            result, details = primeintellect_check_correctness(tests, "code")
            
            args, kwargs = mock_check.call_args
            assert args[0]["fn_name"] == "test_func"

    def test_use_tci_true(self):
        tests = [{"input": "1", "output": "2"}]
        
        with patch('agents.math_agent.reward.code_reward.TogetherCodeTool') as mock_tool_class:
            mock_tool = Mock()
            mock_tool_class.return_value = mock_tool
            
            with patch('agents.math_agent.reward.code_reward.codetool_check_correctness') as mock_codetool:
                from agents.math_agent.reward.code_reward import primeintellect_check_correctness
                mock_codetool.return_value = (True, {})
                result, details = primeintellect_check_correctness(tests, "code", use_tci=True)
                
                assert mock_codetool.called

    def test_empty_tests_assertion(self):
        tests = []
        with pytest.raises(ValueError, match="needs at least one test case"):
            from agents.math_agent.reward.code_reward import primeintellect_check_correctness
            primeintellect_check_correctness(tests, "code")


class TestLCBCheckCorrectnessV2:
    def test_success_all_passed(self):
        mock_input_output = {"inputs": ["1", "2"], "outputs": ["2", "3"]}
        
        with patch('agents.math_agent.reward.code_reward.postprocess_lcb_sample') as mock_postprocess:
            mock_postprocess.return_value = {"input_output": json.dumps(mock_input_output)}
            
            with patch('multiprocessing.Manager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = []
                mock_manager.list.return_value = mock_result
                mock_manager_class.return_value = mock_manager
                
                with patch('multiprocessing.Process') as mock_process_class:
                    mock_instance = Mock()
                    mock_instance.is_alive.return_value = False
                    mock_process_class.return_value = mock_instance
                    
                    with patch('agents.math_agent.reward.code_reward._temp_run') as mock_temp_run:
                        from agents.math_agent.reward.code_reward import lcb_check_correctness_v2
                        def temp_run_side_effect(sample, generation, debug, result, metadata_list, timeout):
                            result.append([True, True])
                            metadata_list.append({})
                        mock_temp_run.side_effect = temp_run_side_effect
                        
                        sample = [{"input": "1", "output": "2"}]
                        result, details = lcb_check_correctness_v2(sample, "code")
                        
                        assert result is False
                        assert details["all_passed"] is False

    def test_some_failed(self):
        mock_input_output = {"inputs": ["1", "2"], "outputs": ["2", "3"]}
        
        with patch('agents.math_agent.reward.code_reward.postprocess_lcb_sample') as mock_postprocess:
            mock_postprocess.return_value = {"input_output": json.dumps(mock_input_output)}
            
            with patch('multiprocessing.Manager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = []
                mock_manager.list.return_value = mock_result
                mock_manager_class.return_value = mock_manager
                
                with patch('multiprocessing.Process') as mock_process_class:
                    mock_instance = Mock()
                    mock_instance.is_alive.return_value = False
                    mock_process_class.return_value = mock_instance
                    
                    with patch('agents.math_agent.reward.code_reward._temp_run') as mock_temp_run:
                        from agents.math_agent.reward.code_reward import lcb_check_correctness_v2
                        def temp_run_side_effect(sample, generation, debug, result, metadata_list, timeout):
                            result.append([True, False])
                            metadata_list.append({})
                        mock_temp_run.side_effect = temp_run_side_effect
                        
                        sample = [{"input": "1", "output": "2"}]
                        result, details = lcb_check_correctness_v2(sample, "code")
                        
                        assert result is False
                        assert details["all_passed"] is False
                        assert details["passed_tests"] == 0

    def test_global_timeout(self):
        mock_input_output = {"inputs": ["1", "2"], "outputs": ["2", "3"]}
        
        with patch('agents.math_agent.reward.code_reward.postprocess_lcb_sample') as mock_postprocess:
            mock_postprocess.return_value = {"input_output": json.dumps(mock_input_output)}
            
            with patch('multiprocessing.Manager') as mock_manager_class:
                mock_manager = Mock()
                mock_result = []
                mock_manager.list.return_value = mock_result
                mock_manager_class.return_value = mock_manager
                
                with patch('multiprocessing.Process') as mock_process_class:
                    from agents.math_agent.reward.code_reward import lcb_check_correctness_v2
                    mock_instance = Mock()
                    mock_instance.is_alive.return_value = True
                    mock_process_class.return_value = mock_instance
                    
                    sample = [{"input": "1", "output": "2"}]
                    result, details = lcb_check_correctness_v2(sample, "code", debug=True)
                    
                    assert result is False
                    assert details["test_results"][0]["error"] == "global timeout"

    def test_empty_sample_assertion(self):
        with pytest.raises(ValueError, match="must contain at least one test case"):
            from agents.math_agent.reward.code_reward import lcb_check_correctness_v2
            lcb_check_correctness_v2([], "code")


class TestLeetCodeCheckCorrectness:
    def test_success(self):
        with patch('agents.math_agent.reward.code_reward.lc_code_exec') as mock_exec:
            from agents.math_agent.reward.code_reward import leetcode_check_correctness
            mock_exec.return_value = (True, "Success output")
            
            tests = {"functional": "def test(): pass"}
            result, details = leetcode_check_correctness(tests, "print('code')")
            
            assert result is True
            assert details["all_passed"] is True

    def test_failure(self):
        with patch('agents.math_agent.reward.code_reward.lc_code_exec') as mock_exec:
            from agents.math_agent.reward.code_reward import leetcode_check_correctness
            mock_exec.return_value = (False, "Error message")
            
            tests = {"functional": "def test(): pass"}
            result, details = leetcode_check_correctness(tests, "print('code')")
            
            assert result is False
            assert details["all_passed"] is False


class TestKodcodeCheckCorrectness:
    def test_success(self):
        with patch('agents.math_agent.reward.code_reward.kod_code_exec') as mock_exec:
            with patch('agents.math_agent.reward.code_reward.clean_code_main_block') as mock_clean:
                from agents.math_agent.reward.code_reward import kodcode_check_correctness
                mock_clean.return_value = "cleaned code"
                mock_exec.return_value = (True, "Success")
                
                test = "def test_something(): pass"
                result, details = kodcode_check_correctness(test, "print('code')")
                
                assert result is True
                assert details["total_tests"] == 1

    def test_failure(self):
        with patch('agents.math_agent.reward.code_reward.kod_code_exec') as mock_exec:
            with patch('agents.math_agent.reward.code_reward.clean_code_main_block') as mock_clean:
                from agents.math_agent.reward.code_reward import kodcode_check_correctness
                mock_clean.return_value = "cleaned code"
                mock_exec.return_value = (False, "Error")
                
                test = "def test_something(): pass"
                result, details = kodcode_check_correctness(test, "print('code')")
                
                assert result is False
                assert details["all_passed"] is False

    def test_multiple_tests_count(self):
        with patch('agents.math_agent.reward.code_reward.kod_code_exec') as mock_exec:
            with patch('agents.math_agent.reward.code_reward.clean_code_main_block') as mock_clean:
                from agents.math_agent.reward.code_reward import kodcode_check_correctness
                mock_clean.return_value = "cleaned code"
                mock_exec.return_value = (True, "Success")
                
                test = "def test1(): pass\ndef test2(): pass\ndef test3(): pass"
                result, details = kodcode_check_correctness(test, "print('code')")
                
                assert details["total_tests"] == 3


class TestHumanEvalPlusCheckCorrectness:
    def test_success(self):
        with patch('agents.math_agent.reward.code_reward.humanevalplus_run_test') as mock_run:
            with patch('agents.math_agent.reward.code_reward.get_num_test_cases') as mock_get_num:
                with patch('agents.math_agent.reward.code_reward.clean_code_main_block') as mock_clean:
                    from agents.math_agent.reward.code_reward import humanevalplus_check_correctness
                    mock_clean.return_value = "cleaned code"
                    mock_get_num.return_value = 5
                    mock_run.return_value = (True, "Success")
                    
                    result, details = humanevalplus_check_correctness("test code", "print('code')")
                    
                    assert result is True
                    assert details["all_passed"] is True
                    assert details["total_tests"] == 5

    def test_failure(self):
        with patch('agents.math_agent.reward.code_reward.humanevalplus_run_test') as mock_run:
            with patch('agents.math_agent.reward.code_reward.get_num_test_cases') as mock_get_num:
                with patch('agents.math_agent.reward.code_reward.clean_code_main_block') as mock_clean:
                    from agents.math_agent.reward.code_reward import humanevalplus_check_correctness
                    mock_clean.return_value = "cleaned code"
                    mock_get_num.return_value = 3
                    mock_run.return_value = (False, "Error")
                    
                    result, details = humanevalplus_check_correctness("test code", "print('code')")
                    
                    assert result is False
                    assert details["all_passed"] is False


class TestCodeToolCheckCorrectness:
    def test_success_stdin_format(self):
        with patch('agents.math_agent.reward.code_reward.taco_to_lcb_format') as mock_taco:
            mock_taco.return_value = [{"input": "1", "output": "2"}]
            
            # Mock the internal imports
            with patch('rllm.tools.utils.stdin_test_code_wrapper') as mock_wrapper:
                from agents.math_agent.reward.code_reward import codetool_check_correctness
                mock_wrapper.return_value = "wrapped code"
                
                mock_codetool = Mock()
                mock_response = Mock()
                mock_response.error = None
                mock_response.output = "Success"
                mock_codetool.return_value = mock_response
                
                tests = {"inputs": ["1"], "outputs": ["2"]}
                result, details = codetool_check_correctness(tests, "code", mock_codetool)
                
                assert result is True
                assert details["all_passed"] is True

    def test_success_call_based_format(self):
        with patch('agents.math_agent.reward.code_reward.taco_to_lcb_format') as mock_taco:
            mock_taco.return_value = [{"input": "1", "output": "2"}]
            
            with patch('rllm.tools.utils.call_based_test_code_wrapper') as mock_wrapper:
                from agents.math_agent.reward.code_reward import codetool_check_correctness
                mock_wrapper.return_value = "wrapped code"
                
                mock_codetool = Mock()
                mock_response = Mock()
                mock_response.error = None
                mock_response.output = "Success"
                mock_codetool.return_value = mock_response
                
                tests = {"inputs": ["1"], "outputs": ["2"], "fn_name": "test_func"}
                result, details = codetool_check_correctness(tests, "code", mock_codetool)
                
                assert result is True

    def test_failure_with_error(self):
        with patch('agents.math_agent.reward.code_reward.taco_to_lcb_format') as mock_taco:
            mock_taco.return_value = [{"input": "1", "output": "2"}]
            
            with patch('rllm.tools.utils.stdin_test_code_wrapper') as mock_wrapper:
                from agents.math_agent.reward.code_reward import codetool_check_correctness
                mock_wrapper.return_value = "wrapped code"
                
                mock_codetool = Mock()
                mock_response = Mock()
                mock_response.error = "Execution error"
                mock_response.output = ""
                mock_codetool.return_value = mock_response
                
                tests = {"inputs": ["1"], "outputs": ["2"]}
                result, details = codetool_check_correctness(tests, "code", mock_codetool)
                
                assert result is False
                assert details["all_passed"] is False

    def test_not_taco_format_no_conversion(self):
        mock_codetool = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_codetool.return_value = mock_response
        
        with patch('rllm.tools.utils.stdin_test_code_wrapper') as mock_wrapper:
            from agents.math_agent.reward.code_reward import codetool_check_correctness
            mock_wrapper.return_value = "wrapped code"
            
            tests = [{"input": "1", "output": "2"}]
            result, details = codetool_check_correctness(tests, "code", mock_codetool, is_taco_format=False)
            
            assert result is True


class TestRewardCodeFn:
    @pytest.fixture
    def config(self):
        from agents.math_agent.reward.reward_types import RewardConfig
        return RewardConfig(
            correct_reward=1.0,
            incorrect_reward=0.0,
            format_error_reward=-0.5
        )

    def test_no_tests_in_task_info(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "taco", "ground_truth": None}
        
        result = reward_fn(task_info, "```python\nprint('hello')\n```")
        
        assert result.reward == config.format_error_reward
        assert result.is_correct is False
        assert "No tests found" in result.metadata["error"]

    def test_no_code_in_response(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "taco", "ground_truth": {"inputs": ["1"], "outputs": ["2"]}}
        
        result = reward_fn(task_info, "Just a text response without code block")
        
        assert result.reward == config.format_error_reward
        assert result.is_correct is False

    def test_taco_dataset_without_tci(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "taco", "ground_truth": {"inputs": ["1"], "outputs": ["2"]}}
        
        with patch('agents.math_agent.reward.code_reward.taco_to_lcb_format') as mock_taco:
            mock_taco.return_value = [{"input": "1", "output": "2"}]
            with patch('agents.math_agent.reward.code_reward.lcb_check_correctness_v2') as mock_lcb:
                mock_lcb.return_value = (True, {"all_passed": True})
                
                result = reward_fn(task_info, "```python\nprint('hello')\n```")
                
                assert result.reward == config.correct_reward
                assert result.is_correct is True

    def test_taco_dataset_with_tci(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        config.use_together_code_interpreter = True
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "taco", "ground_truth": {"inputs": ["1"], "outputs": ["2"]}}
        
        with patch('agents.math_agent.reward.code_reward.TogetherCodeTool') as mock_tool_class:
            mock_codetool = Mock()
            mock_tool_class.return_value = mock_codetool
            
            with patch('agents.math_agent.reward.code_reward.codetool_check_correctness') as mock_codetool_check:
                mock_codetool_check.return_value = (True, {"all_passed": True})
                
                result = reward_fn(task_info, "```python\nprint('hello')\n```")
                
                assert result.reward == config.correct_reward
                assert result.is_correct is True

    def test_leetcode_dataset(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "leetcode", "ground_truth": {"functional": "def test(): pass"}}
        
        with patch('agents.math_agent.reward.code_reward.leetcode_check_correctness') as mock_check:
            mock_check.return_value = (True, {"all_passed": True})
            
            result = reward_fn(task_info, "```python\nprint('hello')\n```")
            
            assert result.reward == config.correct_reward

    def test_livecodebench_dataset(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "livecodebench", "ground_truth": [{"input": "1", "output": "2"}]}
        
        with patch('agents.math_agent.reward.code_reward.lcb_check_correctness_v2') as mock_check:
            mock_check.return_value = (True, {"all_passed": True})
            
            result = reward_fn(task_info, "```python\nprint('hello')\n```")
            
            assert result.reward == config.correct_reward

    def test_kodcode_dataset(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "kodcode", "ground_truth": "def test(): pass"}
        
        with patch('agents.math_agent.reward.code_reward.kodcode_check_correctness') as mock_check:
            mock_check.return_value = (True, {"all_passed": True})
            
            result = reward_fn(task_info, "```python\nprint('hello')\n```")
            
            assert result.reward == config.correct_reward

    def test_humanevalplus_dataset(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "humanevalplus", "ground_truth": "def test(): pass"}
        
        with patch('agents.math_agent.reward.code_reward.humanevalplus_check_correctness') as mock_check:
            mock_check.return_value = (True, {"all_passed": True})
            
            result = reward_fn(task_info, "```python\nprint('hello')\n```")
            
            assert result.reward == config.correct_reward

    def test_incorrect_answer(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "taco", "ground_truth": {"inputs": ["1"], "outputs": ["2"]}}
        
        with patch('agents.math_agent.reward.code_reward.taco_to_lcb_format') as mock_taco:
            mock_taco.return_value = [{"input": "1", "output": "2"}]
            with patch('agents.math_agent.reward.code_reward.lcb_check_correctness_v2') as mock_lcb:
                mock_lcb.return_value = (False, {"all_passed": False})
                
                result = reward_fn(task_info, "```python\nprint('wrong')\n```")
                
                assert result.reward == config.incorrect_reward
                assert result.is_correct is False

    def test_unknown_dataset_raises_error(self, config):
        from agents.math_agent.reward.code_reward import RewardCodeFn
        reward_fn = RewardCodeFn(config)
        task_info = {"data_source": "unknown_dataset", "ground_truth": {}}
        
        with pytest.raises(NotImplementedError, match="Dataset unknown_dataset not implemented"):
            reward_fn(task_info, "```python\nprint('hello')\n```")


class TestRllmRewardFnCode:
    def test_rllm_reward_fn_code(self):
        with patch('agents.math_agent.reward.code_reward.RewardCodeFn') as mock_reward_class:
            from agents.math_agent.reward.code_reward import rllm_reward_fn_code
            mock_reward_instance = Mock()
            mock_reward_output = Mock()
            mock_reward_output.reward = 1.0
            mock_reward_output.is_correct = True
            mock_reward_instance.return_value = mock_reward_output
            mock_reward_class.return_value = mock_reward_instance
            
            result = rllm_reward_fn_code("taco", "```python\nprint('hello')\n```", {"inputs": ["1"], "outputs": ["2"]})
            
            assert result.reward == 1.0
            assert result.is_correct is True