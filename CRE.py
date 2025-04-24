import pandas as pd
from agents.orchestrator_agent import orchestrator_agent
from agents.sql_agent import sql_agent
from agents.talk_with_user_agent import talk_with_user_agent
from agents.update_plan import update_plan
from agents.create_plan import create_plan
from utils.unique import metadata_str_into_dict, create_filtered_dict # Assuming format_dict_info is not needed for console output
import re
import traceback # For better error reporting
import sys # To exit cleanly on errors

# --- Constants ---
DB_NAME = 'CRE_Data.db'
METADATA_FILE = 'agents/prompts/metadata.txt'

# --- Function Definitions ---

def run_analysis(question, new_data, prior_steps, plan, filtered_dict, db_name):
    """Runs a single step of the analysis using the orchestrator and appropriate tool."""
    try:
        # Call the orchestrator_agent
        print("\n--- Calling Orchestrator ---")
        output = orchestrator_agent(question=question, metadata=filtered_dict, new_data=new_data, prior_steps=prior_steps, current_plan=plan)
        print(f"Orchestrator Output: {output}") # Log orchestrator decision

        tool_to_call = output.get('Call_Tool', {}).get('Tool', 'Unknown Tool')
        instructions = output.get('Call_Tool', {}).get('Instructions', '')

        print(f"Orchestrator decided to call: {tool_to_call}")
        print(f"Instructions: {instructions}")

        # Call the next agent based on orchestrator output
        if tool_to_call == 'Talk with user Tool':
            print("\n--- Calling Talk With User Tool ---")
            message = talk_with_user_agent(
                user_message=instructions,
                metadata=filtered_dict,
                new_data=new_data,
                prior_steps=prior_steps,
                current_plan=plan
            )
            step_output = message.get('Send_To_User', 'No message generated.')
            step_details = f"'Message to User: {step_output}'"
            # Return user message directly
            return tool_to_call, step_output, new_data, f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}", plan

        elif tool_to_call == 'SQL Tool':
            print("\n--- Calling SQL Tool ---")
            # Ensure metadata for the specific DB is passed correctly
            db_metadata = filtered_dict.get(db_name, {})
            if not db_metadata:
                raise ValueError(f"No metadata found for database '{db_name}' in the filtered dictionary.")

            ai_output, updated_new_data = sql_agent(
                instructions,
                metadata = db_metadata,
                database_name = db_name,
                new_data = new_data # Pass and potentially update new_data dict
            )
            # Make sure new_data from the function scope is updated
            new_data = updated_new_data

            df_name = ai_output.get('df_name', 'Unknown DataFrame')
            sql_query = ai_output.get('SQL_Query', 'No SQL Query Provided') # Get SQL query
            print(f"SQL Tool generated DataFrame: {df_name}")
            if sql_query:
                print(f"SQL Query Executed:\n{sql_query}")

            step_details = f"'{instructions}' | OUTPUT DataFrame: {df_name}"

            # Update the plan (optional, depends on update_plan logic)
            print("\n--- Calling Update Plan Tool ---")
            current_step_full_desc = f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}"
            updated_plan = update_plan(prior_steps + [current_step_full_desc], plan)
            print(f"Plan Updated To:\n{updated_plan}")

            return tool_to_call, {'df_name': df_name, 'SQL_Query': sql_query}, new_data, current_step_full_desc, updated_plan

        else:
             # Handle unknown tool case
            print(f"WARNING: Orchestrator requested unknown tool: {tool_to_call}")
            # Decide how to proceed: maybe default to 'Talk with user' or raise error
            # For now, let's signal to stop by pretending it's 'Talk with user' with an error message
            error_message = f"Analysis halted: Orchestrator requested an unknown tool ('{tool_to_call}')."
            step_details = f"'Internal Error: {error_message}'"
            step_description = f"STEP {len(prior_steps) + 1}. [ERROR]: {step_details}"
            return 'Talk with user Tool', error_message, new_data, step_description, plan


    except Exception as e:
        print(f"Error during analysis step: {e}", file=sys.stderr)
        traceback.print_exc()
        # Re-raise or return an error indicator to stop the loop gracefully
        raise e # Stop the execution

# --- Main Execution ---
if __name__ == "__main__":
    print("--- CRE Database Console Runner ---")

    # --- Load Metadata ---
    try:
        print(f"Loading metadata from: {METADATA_FILE}")
        with open(METADATA_FILE, 'r') as file:
            metadata_string = file.read() # Keep the raw string for create_plan
            metadata_dict = metadata_str_into_dict(metadata_string)
            # Get all available tables for the database
            all_available_tables = list(metadata_dict.get(DB_NAME, {}).keys())
            if not all_available_tables:
                print(f"WARNING: No tables found in metadata for database: {DB_NAME}", file=sys.stderr)
            else:
                 print(f"Available tables in {DB_NAME}: {', '.join(all_available_tables)}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Metadata file '{METADATA_FILE}' not found.", file=sys.stderr)
        sys.exit(1) # Stop execution if metadata is missing
    except Exception as e:
        print(f"FATAL ERROR: Error loading or parsing metadata: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1) # Stop execution on other metadata errors

    # --- Get User Input ---
    question = input("Enter your question:\n> ")
    if not question:
        print("No question entered. Exiting.")
        sys.exit(0)

    # --- Planning Phase ---
    plan = ""
    suggested_tables = []
    try:
        print("\n--- Generating Analysis Plan ---")
        # Call create_plan agent
        plan_result, suggested_tables_raw = create_plan(question=question, metadata_string=metadata_string)
        plan = plan_result
        # Validate suggested tables against all available tables
        suggested_tables = [t for t in suggested_tables_raw if t in all_available_tables]

        print(f"\nInitial Plan:\n{plan}")
        print(f"\nSuggested Tables (validated): {', '.join(suggested_tables)}")

        if not suggested_tables:
             print(f"WARNING: The planner suggested tables ({', '.join(suggested_tables_raw)}), but none are available in the metadata for {DB_NAME}. Analysis might fail.", file=sys.stderr)
             # Decide if you want to proceed or exit. For now, let's try proceeding.
             # If proceeding without tables is impossible, uncomment the next lines:
             # print("Cannot proceed without valid tables. Exiting.")
             # sys.exit(1)

    except Exception as e:
        print(f"\nERROR during planning phase: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1) # Exit if planning fails


    # --- Analysis Execution (using suggested tables directly) ---
    print(f"\n--- Starting Analysis Using Tables: {', '.join(suggested_tables)} ---")

    # Create filtered metadata based on the *suggested* selection
    try:
        filtered_metadata_dict = create_filtered_dict(metadata_dict, suggested_tables, db_name=DB_NAME)
        # Optional: Print the filtered metadata being used
        # print(f"\nFiltered Metadata for Analysis:\n{filtered_metadata_dict}")
    except Exception as e:
        print(f"\nERROR: Failed to filter metadata based on suggested tables: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize state for the analysis run
    prior_steps_list = []
    current_new_data = {} # Stores generated dataframes {name: {'data_frame': df, ...}}
    analysis_plan = plan # Use the plan generated earlier
    talk_to_user_flag = False
    final_output = "Analysis could not be completed." # Default final message
    analysis_steps_details = [] # Store details like SQL for final summary

    try:
        print("\n--- Running Analysis Steps ---")
        # The main analysis loop
        while not talk_to_user_flag:
            print(f"\nExecuting Step {len(prior_steps_list) + 1}...")
            # Call the run_analysis function for one step
            tool_called, output, current_new_data, step_description, analysis_plan = run_analysis(
                question,              # Pass the original question
                current_new_data,      # Pass the current data dictionary
                prior_steps_list,      # Pass the list of prior steps descriptions
                analysis_plan,         # Pass the current plan
                filtered_metadata_dict,# Pass the filtered metadata
                DB_NAME                # Pass the database name
            )

            # Store step details
            step_info = {"step": step_description, "tool": tool_called}
            if tool_called == 'SQL Tool' and isinstance(output, dict):
                step_info["sql"] = output.get('SQL_Query')
            analysis_steps_details.append(step_info)

            # Append step description for the next iteration's input
            prior_steps_list.append(step_description)

            # Print step summary to console
            print(f"Step Description: {step_description}")
            if step_info.get("sql"):
                print(f"SQL Executed in step: \n{step_info['sql']}")

            # Check if the loop should terminate
            if tool_called == 'Talk with user Tool':
                talk_to_user_flag = True
                final_output = output # The output is the message string from talk_with_user_agent or an error

        # --- Analysis Finished ---
        print("\n--- Analysis Loop Completed ---")

    except Exception as e:
        # Error occurred during the loop execution
        print(f"\nFATAL ERROR during the analysis execution loop: {e}", file=sys.stderr)
        traceback.print_exc()
        final_output = f"Analysis stopped due to error: {e}"
        # Keep partial results stored in current_new_data and analysis_steps_details

    # --- Results Display Section ---
    print("\n\n========================================")
    print("         ANALYSIS RESULTS")
    print("========================================")

    # Display steps executed
    print("\n--- Analysis Steps Executed ---")
    if not analysis_steps_details:
        print("No analysis steps were executed.")
    else:
        for item in analysis_steps_details:
            print(item.get("step", "Step details missing"))
            if item.get("tool") == 'SQL Tool' and item.get("sql"):
                print(f"  SQL Query:\n{item['sql']}")
            print("-" * 20) # Separator

    # Display final message
    print("\n--- Final Analysis Result ---")
    print(final_output) # Display the final message

    # Find and display dataframe previews
    print("\n--- Generated DataFrames (if any) ---")
    generated_data = current_new_data
    matches = re.findall(r'\[\[(.*?)\]\]', final_output) # Find [[df_name]] mentions in the *final* output
    displayed_dfs = set()

    if not generated_data:
        print("No DataFrames were generated during the analysis.")
    elif not matches:
        print("Final result message does not explicitly mention any DataFrames ([[df_name]]).")
        # Optional: List all generated DFs anyway
        print("Listing all generated DataFrames:")
        for name, df_info in generated_data.items():
             df = df_info.get('data_frame')
             if isinstance(df, pd.DataFrame):
                  print(f"\n** DataFrame: `{name}` ** (Preview)")
                  print(df.head().to_string()) # Print head() to console
                  displayed_dfs.add(name)
    else:
        # Process DFs mentioned in the final output
        print("Displaying DataFrames mentioned in the final result:")
        for name in matches:
            name = name.strip() # Clean whitespace
            if name in generated_data and name not in displayed_dfs:
                df_info = generated_data[name]
                # Assuming structure is {'df_name': {'data_frame': pd.DataFrame, ...}}
                df = df_info.get('data_frame')

                if isinstance(df, pd.DataFrame):
                    print(f"\n** DataFrame: `{name}` ** (Preview)")
                    # Print DataFrame head to console using to_string() for better formatting
                    try:
                        print(df.head().to_string())
                    except Exception as df_print_err:
                         print(f"  Could not print preview for {name}: {df_print_err}")
                    displayed_dfs.add(name) # Mark as displayed
                else:
                    print(f"\nWARNING: Data for `[[{name}]]` found but is not a valid DataFrame.")
                    displayed_dfs.add(name) # Avoid re-warning

            elif name not in generated_data and name not in displayed_dfs:
                # Only warn if mentioned but not found AND not already warned about
                print(f"\nWARNING: DataFrame `[[{name}]]` mentioned in the result but was not found in the generated data.")
                displayed_dfs.add(name) # Avoid re-warning

        # Optional: Check if any generated DFs were *not* mentioned
        unmentioned_dfs = set(generated_data.keys()) - displayed_dfs
        if unmentioned_dfs:
            print("\n--- Additional Generated DataFrames (not mentioned in final result) ---")
            for name in unmentioned_dfs:
                 df_info = generated_data[name]
                 df = df_info.get('data_frame')
                 if isinstance(df, pd.DataFrame):
                    print(f"\n** DataFrame: `{name}` ** (Preview)")
                    try:
                         print(df.head().to_string())
                    except Exception as df_print_err:
                         print(f"  Could not print preview for {name}: {df_print_err}")

    print("\n--- End of Execution ---")
