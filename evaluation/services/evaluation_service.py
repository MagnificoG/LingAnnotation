import pandas as pd
import ast
import logging
import asyncio
from asgiref.sync import sync_to_async
from openai import AsyncOpenAI, APIError, RateLimitError
import os
import json # For JSON input/output
from tqdm.asyncio import tqdm as async_tqdm # For progress bar
import datetime # For timestamped output files
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Data Loading ---

class DataLoader:
    """Loads data from the source task's data.json file."""

    def __init__(self, task_id: int, task_dirpath: str):
        """
        Initializes the DataLoader with the necessary path information.

        Args:
            task_id (int): The ID of the evaluation task for logging.
            task_dirpath (str): The absolute path to the source task's directory.
        """
        self.task_id = task_id
        self.task_dirpath = task_dirpath

    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from the source task's data.json file."""
        logging.info(f"Attempting to load data for task {self.task_id} from directory {self.task_dirpath}")
        try:
            # Correctly construct the path from the provided directory
            data_filepath = Path(self.task_dirpath) / 'data.json'
            with data_filepath.open("r", encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded data for task {self.task_id}")
            return data
        except FileNotFoundError:
            logging.error(f"The file 'data.json' was not found in {self.task_dirpath}", exc_info=True)
            raise  # Re-raise to be caught by the runner
        except Exception as e:
            logging.error(f"Error loading data for task {self.task_id}: {str(e)}", exc_info=True)
            raise  # Re-raise the exception to be caught by the runner

# --- 2. Data Transformation ---

class DataTransformer:
    """Transforms raw data from dataset items into structured task dictionaries."""
    
    def transform(self, data):
        """
        Transforms the data into a list of structured task dictionaries.
        
        Args:
            data (List[Dict]): The input data loaded by DataLoader.
            
        Returns:
            List[Dict[str, Any]]: A list where each dictionary represents a task
                                in the desired format. Returns empty list on failure.
        """
        if not data:
            logging.warning("Input data is empty or None. Cannot transform data.")
            return []
        
        processed_tasks = []
        
        for item in data:
            task_dict = {
                'text': item.get('text', ''),
                'ground_truth': item.get('labels', []),  # Use labels as ground truth
                'id': item.get('id')
            }
            processed_tasks.append(task_dict)
            logging.debug(f"Successfully transformed item {task_dict.get('id', 'N/A')}")
        
        logging.info(f"Transformation complete. Processed {len(processed_tasks)} tasks.")
        return processed_tasks

# --- 3. LLM Interaction ---

class AsyncLLMClient:
    """
    ASYNC Client for interacting with an LLM API (OpenAI compatible).
    Handles prompt formatting, API calls, and basic response parsing asynchronously.
    """
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        """
        Initializes the Async LLM client.

        Args:
            model_name (str): Identifier for the LLM model.
            api_key (str): The API key for the service.
            base_url (Optional[str]): The base URL for the API (e.g., for DeepSeek or local models).
                                     If None, uses the default OpenAI URL.
        """
        self.model_name = model_name
        # Allow omitting base_url for default OpenAI
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        try:
            self.client = AsyncOpenAI(**client_args)
            url_info = f"at {base_url}" if base_url else "at default OpenAI URL"
            logging.info(f"AsyncLLMClient initialized for model: {self.model_name} {url_info}")
        except Exception as e:
            logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
            raise

    # --- Prompt Formatting (Synchronous Helpers) ---
    def _format_options(self, options: Optional[Dict]) -> str:
        if not options or not isinstance(options, dict): return "No options provided."
        return "\n".join([f"{key}: {value}" for key, value in options.items()])

    def _format_prompt(self, task_data: Dict[str, Any]) -> List[Dict[str, str]]:
        system_message = (
            "You are an AI assistant evaluating language tasks. "
            "Follow the instructions precisely. "
            "Provide only the answer in the format requested. "
            "Do not add explanations unless explicitly asked."
            # Add specific format hints if useful
        )
        user_prompt_parts = []
        # 获取指令并添加到提示部分
        inst = task_data.get('instruction')
        if inst:
            user_prompt_parts.append(f"Instruction: {inst}")
        text = task_data.get('text')
        if text:
            if str(text).strip(): user_prompt_parts.append("\n--- Context ---\n" + str(text))
        question = task_data.get('question')
        if question: user_prompt_parts.append("\n--- Question ---\n" + str(question))
        options = task_data.get('options')
        if options and isinstance(options, dict) and options:
            user_prompt_parts.append("\n--- Options ---\n" + self._format_options(options))
        user_prompt_parts.append("\n--- Answer ---")
        user_prompt = "\n".join(user_prompt_parts)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]
        # logging.debug(f"Formatted messages for LLM task {task_data.get('id', 'N/A')}: {messages}")
        return messages

    # --- Response Parsing (Synchronous Helper) ---
    def _parse_llm_response(self, response_text: str, task_id: str = "N/A") -> Any:
        cleaned_response = response_text.strip()
        if not cleaned_response:
            logging.warning(f"Task {task_id} (Model {self.model_name}): Received empty response string.")
            return None

        try:
            if (cleaned_response.startswith('[') and cleaned_response.endswith(']')) or \
               (cleaned_response.startswith('{') and cleaned_response.endswith('}')):
                parsed = ast.literal_eval(cleaned_response)
                if isinstance(parsed, list): return parsed # Primarily expect lists
                logging.warning(f"Task {task_id} (Model {self.model_name}): Parsed non-list literal {type(parsed)}. Returning as is.")
                return parsed # Return dicts, etc., but evaluator might not handle
            elif cleaned_response.lower() in ['true', 'false', 'none']:
                return ast.literal_eval(cleaned_response.lower().capitalize())
            # Stricter number check
            elif (cleaned_response.startswith('-') and cleaned_response[1:].replace('.', '', 1).isdigit()) or \
                 (cleaned_response.replace('.', '', 1).isdigit()):
                 return float(cleaned_response) if '.' in cleaned_response else int(cleaned_response)
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
             logging.debug(f"Task {task_id} (Model {self.model_name}): Could not parse '{cleaned_response}' as literal ({e}). Treating as string.")
        except Exception as e:
             logging.warning(f"Task {task_id} (Model {self.model_name}): Unexpected parsing error for '{cleaned_response}': {e}. Treating as string.")

        # Default: return string, remove surrounding quotes if present
        if len(cleaned_response) > 1 and \
           ((cleaned_response.startswith('"') and cleaned_response.endswith('"')) or \
            (cleaned_response.startswith("'") and cleaned_response.endswith("'"))):
            return cleaned_response[1:-1]
        return cleaned_response

    # --- Async API Call ---
    async def get_completion(self, task_data: Dict[str, Any]) -> Any:
        """
        ASYNC sends task data to the LLM API and retrieves the completion.

        Args:
            task_data (Dict[str, Any]): The structured task dictionary.

        Returns:
            Any: The parsed response from the LLM, or None if the API call
                 fails or returns an empty/malformed response.
        """
        task_id = str(task_data.get('id', 'N/A'))
        logging.debug(f"Task {task_id}: Requesting completion from {self.model_name}")

        messages = self._format_prompt(task_data)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=250, # Adjust as needed
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                raw_response = response.choices[0].message.content
                logging.debug(f"Task {task_id} (Model {self.model_name}): Raw response: '{raw_response}'")
                parsed_response = self._parse_llm_response(raw_response, task_id)
                logging.debug(f"Task {task_id} (Model {self.model_name}): Parsed response: {parsed_response} (Type: {type(parsed_response)})")
                return parsed_response
            else:
                logging.warning(f"Task {task_id} (Model {self.model_name}): LLM API returned empty/malformed choices. Response: {response}")
                return None

        except RateLimitError as e:
            logging.error(f"Task {task_id} (Model {self.model_name}): Rate limit error. {e}. Consider adding delays or backoff.")
            # TODO: Implement retry/backoff logic here if needed
            await asyncio.sleep(5) # Simple delay as placeholder
            return None # Indicate failure after potential delay/retry
        except APIError as e:
            logging.error(f"Task {task_id} (Model {self.model_name}): API Error (Status: {e.status_code}, Type: {e.type}): {e.message}")
            return None
        except Exception as e:
            logging.error(f"Task {task_id} (Model {self.model_name}): Unexpected error during API call: {type(e).__name__} - {e}")
            return None

# --- 3.5 LLM Provider Management ---

class LLMProvider:
    """Encapsulates information and the client for a specific LLM provider and model."""
    def __init__(self, provider_name: str, model_name: str, api_key: str, base_url: Optional[str] = None):
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        # Instantiate the async client specific to this provider config
        self.client = AsyncLLMClient(model_name=model_name, api_key=api_key, base_url=base_url)

    def get_identifier(self) -> str:
        """Returns a unique string identifier for this provider/model config."""
        return f"{self.provider_name}__{self.model_name}"

# --- 4. Evaluation ---

class Evaluator:
    """Compares the LLM's response against the ground truth answer."""

    # CORRECTED SIGNATURE: Added task_id and provider_id with defaults
    def evaluate(self, llm_response: Any, ground_truth: Any, task_id: str = "N/A", provider_id: str = "N/A") -> bool:
        """
        Compares the LLM response with the ground truth answer. Handles None gracefully.

        Args:
            llm_response (Any): The response obtained from the LLMClient.
            ground_truth (Any): The parsed ground truth answer from the data.
            task_id (str): Optional task ID for logging context.
            provider_id (str): Optional provider ID for logging context.

        Returns:
            bool: True if the response matches the ground truth, False otherwise.
        """
        # Use the provided IDs in logging messages
        log_prefix = f"Task {task_id} ({provider_id})"
        logging.debug(f"{log_prefix}: Evaluating LLM Response: {llm_response} (Type: {type(llm_response)}) vs GT: {ground_truth} (Type: {type(ground_truth)})")

        # Handle cases where one or both are None
        if ground_truth is None:
             is_correct = llm_response is None
             logging.warning(f"{log_prefix}: Ground truth is None. Evaluation result based on LLM response being None: {is_correct}")
             return is_correct
        if llm_response is None:
             logging.info(f"{log_prefix}: LLM response is None (API error/empty), but ground truth exists. Considered incorrect.")
             return False

        # --- Comparison Logic (Types should ideally match after parsing) ---

        # Case 1: Ground truth is a list
        if isinstance(ground_truth, list):
            if not isinstance(llm_response, list):
                logging.warning(f"{log_prefix}: Eval type mismatch: GT is list, LLM response is {type(llm_response)} ('{llm_response}'). Incorrect.")
                return False
            try:
                # Compare content ignoring order and simple type differences (str vs int)
                gt_list_str = sorted([str(item).strip() for item in ground_truth])
                resp_list_str = sorted([str(item).strip() for item in llm_response])
                is_correct = resp_list_str == gt_list_str and len(resp_list_str) == len(gt_list_str) # Also check length
                logging.debug(f"{log_prefix}: List comparison: sorted_str({llm_response}) == sorted_str({ground_truth}) -> {is_correct}")
                return is_correct
            except Exception as e: # Catch potential comparison errors
                 logging.warning(f"{log_prefix}: Error comparing lists '{llm_response}' vs '{ground_truth}': {e}. Using direct equality.")
                 return llm_response == ground_truth # Fallback

        # Case 2: Ground truth is a boolean
        elif isinstance(ground_truth, bool):
            if not isinstance(llm_response, bool):
                 logging.warning(f"{log_prefix}: Eval type mismatch: GT is bool, LLM response is {type(llm_response)} ('{llm_response}'). Incorrect.")
                 return False
            is_correct = llm_response == ground_truth
            logging.debug(f"{log_prefix}: Boolean comparison: {llm_response} == {ground_truth} -> {is_correct}")
            return is_correct

        # Case 3: Ground truth is a number (int/float)
        elif isinstance(ground_truth, (int, float)):
             if not isinstance(llm_response, (int, float)):
                 logging.warning(f"{log_prefix}: Eval type mismatch: GT is number, LLM response is {type(llm_response)} ('{llm_response}'). Incorrect.")
                 return False
             try:
                # Use tolerance for float comparison
                is_correct = abs(float(llm_response) - float(ground_truth)) < 1e-9
                logging.debug(f"{log_prefix}: Numeric comparison: {float(llm_response)} ≈ {float(ground_truth)} -> {is_correct}")
                return is_correct
             except Exception as e:
                 logging.error(f"{log_prefix}: Error during numeric comparison: {e}")
                 return False

        # Case 4: Ground truth is a string (default/fallback)
        else:
            llm_response_str = str(llm_response).strip()
            ground_truth_str = str(ground_truth).strip()
            # is_correct = llm_response_str.lower() == ground_truth_str.lower() # Case-insensitive option
            is_correct = llm_response_str == ground_truth_str # Case-sensitive
            logging.debug(f"{log_prefix}: String comparison: '{llm_response_str}' == '{ground_truth_str}' -> {is_correct}")
            return is_correct

# --- 5. Orchestration ---

class EvaluationRunner:
    """
    Orchestrates async evaluation using multiple LLM providers,
    with checkpointing and results persistence.
    """
    def __init__(self, data_loader: DataLoader, data_transformer: DataTransformer,
                 providers: List[LLMProvider], # Accepts list of providers
                 evaluator: Evaluator,
                 output_json_path: Optional[str] = None,
                 checkpoint_interval: int = 20, # Checkpoint frequency
                 concurrency_limit: int = 10): # Max parallel requests
        """
        Initializes the Async EvaluationRunner.

        Args:
            data_loader: Instance of DataLoader.
            data_transformer: Instance of DataTransformer.
            providers: List of configured LLMProvider instances.
            evaluator: Instance of Evaluator.
            output_json_path (Optional[str]): Path to save the results JSON file.
            checkpoint_interval (int): Save after every N completed results.
            concurrency_limit (int): Max number of concurrent API calls.
        """
        if not providers:
            raise ValueError("At least one LLMProvider must be specified.")

        self.data_loader = data_loader
        self.data_transformer = data_transformer
        self.providers = providers
        self.evaluator = evaluator
        self.checkpoint_interval = max(1, checkpoint_interval) # Ensure at least 1
        self.semaphore = asyncio.Semaphore(concurrency_limit) # Control concurrency

        # Determine output path
        if output_json_path:
            self.output_json_path = output_json_path
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include provider names in default filename if multiple, keep simple if one
            provider_slug = providers[0].get_identifier() if len(providers) == 1 else f"{len(providers)}providers"
            self.output_json_path = f"evaluation_results_{provider_slug}_{timestamp}.json"
            logging.info(f"No output path specified, using default: {self.output_json_path}")

        self.all_tasks_data: List[Dict[str, Any]] = [] # All tasks from Excel
        self.results: List[Dict[str, Any]] = [] # Holds results (loaded + new)
        # Tracks completed (task_id, provider_identifier) tuples
        self.processed_combinations: Set[Tuple[str, str]] = set()


    def _load_previous_results(self):
        """Loads results and populates processed_combinations set."""
        self.results = []
        self.processed_combinations = set()
        if os.path.exists(self.output_json_path):
            logging.info(f"Found existing results file: {self.output_json_path}. Loading.")
            try:
                with open(self.output_json_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                     valid_results = []
                     for res in loaded_data:
                         # Validate essential keys for resumption
                         task_id = res.get('task_id')
                         provider_id = res.get('provider_identifier') # Use combined identifier
                         if task_id and provider_id:
                             valid_results.append(res)
                             self.processed_combinations.add((str(task_id), str(provider_id)))
                         else:
                             logging.warning(f"Skipping loaded result due to missing 'task_id' or 'provider_identifier': {res}")
                     self.results = valid_results
                     logging.info(f"Loaded {len(self.results)} valid previous results covering {len(self.processed_combinations)} task/provider combinations.")
                else:
                     logging.warning(f"Loaded data from {self.output_json_path} is not a list. Starting fresh.")
            except Exception as e:
                logging.error(f"Error loading/parsing results file {self.output_json_path}: {e}. Starting fresh.")
                self.results = []
                self.processed_combinations = set()
        else:
            logging.info("No previous results file found. Starting fresh.")

    def _save_results_to_json(self):
        """Saves the current cumulative results list to the output JSON file."""
        logging.debug(f"Saving {len(self.results)} results to {self.output_json_path}")
        try:
            os.makedirs(os.path.dirname(self.output_json_path) or '.', exist_ok=True)
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)
            logging.debug(f"Progress saved successfully to {self.output_json_path}")
        except Exception as e:
            logging.error(f"Error saving results to {self.output_json_path}: {e}")

    async def _get_completion_and_evaluate(self, provider: LLMProvider, task_data: Dict[str, Any]):
        """Coroutine wrapper to get completion, evaluate, and return structured result."""
        task_id = str(task_data.get('id'))
        provider_id = provider.get_identifier()

        # Acquire semaphore to limit concurrency
        async with self.semaphore:
            llm_response = await provider.client.get_completion(task_data)

        # Evaluation happens outside the semaphore lock
        ground_truth = task_data.get('ground_truth')
        is_correct = None
        if ground_truth is not None:
            is_correct = self.evaluator.evaluate(llm_response, ground_truth, task_id, provider_id)

        result = {
            "task_id": task_id,
            "provider_identifier": provider_id, # Store combined ID
            "provider_name": provider.provider_name,
            "model_name": provider.model_name,
            "llm_response": llm_response,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "task_instruction": task_data.get('instruction'), # Include key parts of task for context
            "task_question": task_data.get('question'),
             # Optionally include full task_data if needed, but makes JSON large
             # "full_task_data": task_data,
        }
        return result


    async def _process_tasks_async(self) -> List[Dict[str, Any]]:
        """The core async task processing loop."""

        tasks_to_create = []
        # Identify which (task, provider) combinations need processing
        for task_data in self.all_tasks_data:
            task_id = str(task_data.get('id'))
            for provider in self.providers:
                provider_id = provider.get_identifier()
                if (task_id, provider_id) not in self.processed_combinations:
                    # Create the coroutine object, but don't await yet
                    # Pass provider and task_data needed by the wrapper
                    coro = self._get_completion_and_evaluate(provider, task_data)
                    tasks_to_create.append(coro)

        if not tasks_to_create:
            logging.info("No new task/provider combinations to process based on loaded results.")
            return self.results # Return existing results

        logging.info(f"Creating {len(tasks_to_create)} new API call/evaluation tasks...")

        new_results_count = 0
        # Use asyncio.as_completed for progress tracking as tasks finish
        # Wrap with async_tqdm for the progress bar
        future_iterator = asyncio.as_completed(tasks_to_create)
        progress_bar = async_tqdm(future_iterator, total=len(tasks_to_create), desc="Evaluating Tasks", unit="task")

        for future in progress_bar:
            try:
                # Result is the dictionary returned by _get_completion_and_evaluate
                result_detail = await future
                if result_detail: # Ensure result is not None
                     self.results.append(result_detail)
                     # Add to processed set immediately after successful completion
                     self.processed_combinations.add((result_detail['task_id'], result_detail['provider_identifier']))
                     new_results_count += 1

                     # Checkpoint saving based on count of *newly completed* results
                     if new_results_count % self.checkpoint_interval == 0:
                         self._save_results_to_json()

            except Exception as e:
                # Exceptions from _get_completion_and_evaluate (should be rare if handled internally)
                logging.error(f"Error processing completed task future: {type(e).__name__} - {e}")
                # Optionally: Add more robust error tracking here

        # Final save after the loop
        if new_results_count > 0:
             logging.info("Saving final results...")
             self._save_results_to_json()

        return self.results


    async def run_evaluation(self) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
        """
        Asynchronous entry point that loads data and runs the async processing.

        Returns:
            Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
                - Dictionary of accuracy per provider identifier, or None.
                - The final list of detailed results.
        """
        # 1. Load & Transform Data (still synchronous)
        try:
            raw_data = self.data_loader.load_data()
        except Exception as e:
            # Use task_id from data_loader for logging
            logging.error(f"Error loading raw_data for task {self.data_loader.task_id}: {str(e)}")
            return None, []
        if raw_data is None:
            logging.error("Failed to load data, stopping evaluation.")
            return None, []
        
        self.all_tasks_data = self.data_transformer.transform(raw_data)
        if not self.all_tasks_data:
            logging.warning("Data transformation resulted in no tasks.")
            return None, []

        # 2. Load Previous Results
        self._load_previous_results()

        # 3. Run Async Processing Loop
        # This is now awaited by the caller's event loop
        final_results = await self._process_tasks_async()
        self.results = final_results  # Update self.results with the final list

        # 4. Calculate Final Accuracy (Per Provider)
        provider_stats = {provider.get_identifier(): {'correct': 0, 'evaluated': 0} for provider in self.providers}
        for result in self.results:
             provider_id = result.get('provider_identifier')
             if provider_id in provider_stats:
                 # Only count if evaluation happened (is_correct is not None)
                 if result.get('is_correct') is not None:
                     provider_stats[provider_id]['evaluated'] += 1
                     if result.get('is_correct') is True:
                         provider_stats[provider_id]['correct'] += 1

        accuracies = {}
        logging.info(f"--- Evaluation Summary ---")
        logging.info(f"Total unique tasks in dataset: {len(self.all_tasks_data)}")
        logging.info(f"Total results generated (cumulative): {len(self.results)}")

        for provider_id, stats in provider_stats.items():
            evaluated_count = stats['evaluated']
            correct_count = stats['correct']
            accuracy = None
            if evaluated_count > 0:
                accuracy = correct_count / evaluated_count
                accuracies[provider_id] = accuracy
                logging.info(f"  Provider {provider_id}: Accuracy = {accuracy:.4f} ({correct_count}/{evaluated_count} evaluated tasks)")
            else:
                 logging.info(f"  Provider {provider_id}: No tasks evaluated.")

        logging.info(f"Detailed results saved to: {self.output_json_path}")

        return accuracies if accuracies else None, self.results