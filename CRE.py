import streamlit as st
import pandas as pd
from agents.orchestrator_agent import orchestrator_agent
from agents.sql_agent import sql_agent
from agents.talk_with_user_agent import talk_with_user_agent
from agents.update_plan import update_plan
from agents.create_plan import create_plan
from utils.unique import metadata_str_into_dict
import re
import traceback # For better error reporting

DB_NAME = 'CRE_Data.db'

# --- Initialize Session State ---
# Persists values across reruns/button clicks
if 'planning_complete' not in st.session_state:
    st.session_state.planning_complete = False # Flag to show checklist/run button
if 'plan' not in st.session_state:
    st.session_state.plan = "" # Stores the plan generated
if 'suggested_tables' not in st.session_state:
    st.session_state.suggested_tables = [] # Stores tables suggested by create_plan
# confirmed_tables will be implicitly stored by the multiselect widget's key or read directly
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None # Stores the final output message
if 'new_data_results' not in st.session_state:
      st.session_state.new_data_results = {} # Stores generated dataframes {name: {'data_frame': df, ...}}
if 'prior_steps_display' not in st.session_state:
    st.session_state.prior_steps_display = [] # Stores steps for sidebar display
# Initialize user question state ONCE
if 'user_question_input' not in st.session_state:
    st.session_state.user_question_input = ""

# --- Load Metadata ---
# Load once and store potentially in session state or rely on Streamlit's script execution model
try:
    with open('agents/prompts/metadata.txt', 'r') as file:
        metadata_string = file.read() # Keep the raw string for create_plan
        metadata_dict = metadata_str_into_dict(metadata_string)

        # Get all available tables for the database
        all_available_tables = list(metadata_dict.keys())
        if not all_available_tables:
              st.warning(f"No tables found in metadata for database: {DB_NAME}")
except FileNotFoundError:
    st.error("Metadata file 'agents/prompts/metadata.txt' not found. Please ensure the file exists.")
    st.stop() # Stop execution if metadata is missing
except Exception as e:
    st.error(f"Error loading or parsing metadata: {e}")
    st.error(traceback.format_exc())
    st.stop() # Stop execution on other metadata errors

# --- Function Definitions ---

# run_analysis function remains the same as before
def run_analysis(question, new_data, prior_steps, plan, filtered_dict):
    """Runs a single step of the analysis using the orchestrator and appropriate tool."""
    try:
        # Call the orchestrator_agent
        output = orchestrator_agent(question=question, metadata=filtered_dict, new_data=new_data, prior_steps=prior_steps, current_plan=plan)

        tool_to_call = output.get('Call_Tool', {}).get('Tool', 'Unknown Tool')
        instructions = output.get('Call_Tool', {}).get('Instructions', '')

        # Call the next agent based on orchestrator output
        if tool_to_call == 'Talk with user Tool':
            message = talk_with_user_agent(
                user_message=instructions,
                metadata=filtered_dict,
                new_data=new_data,
                prior_steps=prior_steps,
                current_plan=plan
            )
            step_output = message.get('output', 'No message generated.')
            step = f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: 'Message to User: {step_output}'"

            return tool_to_call, step_output, new_data, step, plan # Return user message directly

        elif tool_to_call == 'SQL Tool':

            ai_output, new_data = sql_agent(
                instructions,
                metadata = filtered_dict,
                database_name = DB_NAME,
                new_data = new_data # Pass and potentially update new_data dict
            )
            df_name = ai_output.get('df_name', 'Unknown DataFrame')
            step = f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: '{instructions}' | OUTPUT DataFrame: {df_name}"
            # Update the plan
            # Ensure prior_steps + [step] is passed correctly if update_plan expects a list
            updated_plan_result = update_plan(prior_steps=prior_steps + [step], current_plan=plan)


            # Store SQL query for display if available
            sql_query = ai_output.get('SQL_Query', None)
            # Make sure to return the updated plan string, handle potential None from update_plan
            current_plan_str = updated_plan_result.get('Plan', plan) if isinstance(updated_plan_result, dict) else plan
            return tool_to_call, {'df_name': df_name, 'SQL_Query': sql_query}, new_data, step, current_plan_str

    except Exception as e:
        st.error(f"Error during analysis step: {e}")
        st.error(traceback.format_exc())
        # Re-raise or return an error indicator to stop the loop gracefully
        raise e # Or return a specific error state


# Caching the CSV conversion function
@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to CSV bytes."""
    try:
        # IMPORTANT: Ensure df is hashable or use a more robust caching strategy if needed
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Error converting DataFrame to CSV: {e}")
        return None

# --- Streamlit App UI ---
st.title("CRE Database Chat")

# Main area for user input - The widget updates st.session_state.user_question_input directly
st.text_area("Enter your question:", key="user_question_input")

# --- Planning Phase ---
col1, col2 = st.columns([1, 3]) # Adjust column widths as needed

with col1:
    if st.button("Plan Analysis", key="plan_button"):
        # Read the question directly from session state (updated by the widget)
        current_question = st.session_state.user_question_input
        if current_question:
            # DO NOT assign back: st.session_state.user_question_input = current_question # REMOVED THIS LINE
            # Reset other state variables for a new planning action
            st.session_state.planning_complete = False
            st.session_state.analysis_results = None
            st.session_state.new_data_results = {}
            st.session_state.prior_steps_display = []

            try:
                with st.spinner("Generating analysis plan..."):
                    # Call create_plan agent using the full metadata initially
                    plan_result, suggested_tables = create_plan(question=current_question, metadata=metadata_dict) # Use current_question
                st.session_state.plan = plan_result['Plan']
                # Validate suggested tables against all available tables
                valid_suggested = [t for t in suggested_tables if t in all_available_tables]
                st.session_state.suggested_tables = valid_suggested
                st.session_state.planning_complete = True
                # Rerun to display the confirmation section immediately
                st.rerun()

            except Exception as e:
                st.error(f"Error during planning phase: {e}")
                st.error(traceback.format_exc())
                st.session_state.planning_complete = False # Ensure flag is false on error
        else:
            st.warning("Please enter a question before planning.")

# --- Table Confirmation and Execution Phase ---
# This section appears only after the 'Plan Analysis' button has been successfully clicked
if st.session_state.planning_complete:
    st.markdown("---") # Visual separator
    st.subheader("Table Selection Confirmation")
    st.markdown("The analysis plan suggests using the tables below. Please confirm or adjust the selection for the analysis:")

    # The multiselect widget manages its own state via the key
    # Use st.session_state.suggested_tables for the default value
    # The selection made by the user is stored in st.session_state.table_confirmation_multiselect
    st.multiselect(
        "Select tables to include in the analysis:",
        options=all_available_tables,
        default=st.session_state.suggested_tables, # Pre-select suggested tables
        key="table_confirmation_multiselect" # Key to access the selection later
    )

    # Add the "Run Analysis" button within this conditional block
    if st.button("Run Analysis with Selected Tables", key="run_analysis_button"):
        # Get the user's selection directly from the widget's state
        selected_tables = st.session_state.table_confirmation_multiselect

        if not selected_tables:
            st.warning("Please select at least one table to run the analysis.")
        else:
            # --- CHECK IF SELECTION MATCHES SUGGESTION ---
            # Use sets for order-independent comparison
            if set(selected_tables) == set(st.session_state.suggested_tables):
                # --- Selection matches: Proceed with Analysis Execution ---
                st.info(f"Starting analysis using tables: `{'`, `'.join(selected_tables)}`")

                # Create filtered metadata based on the user's confirmed selection
                try:
                    filtered_metadata_dict = {table: metadata_dict[table] for table in selected_tables if table in metadata_dict}
                except Exception as e:
                    st.error(f"Failed to filter metadata based on selection: {e}")
                    st.stop()

                # Initialize/reset state for the analysis run
                prior_steps_list = []
                current_new_data = {} # Use a local variable for the loop
                analysis_plan = st.session_state.plan # Get the plan from session state
                talk_to_user_flag = False
                final_output = "Analysis could not be completed." # Default final message

                # Clear previous steps display for this run
                st.session_state.prior_steps_display = []

                try:
                    with st.spinner("Running analysis steps..."):
                        # The main analysis loop
                        step_counter = 0 # Add counter for safety break
                        max_steps = 15 # Define a maximum number of steps
                        while not talk_to_user_flag and step_counter < max_steps :
                            step_counter += 1
                            # Call the run_analysis function for one step
                            # Ensure the current question from session state is used
                            tool_called, output, current_new_data, step_description, analysis_plan = run_analysis(
                                st.session_state.user_question_input,  # Read question from session state
                                current_new_data,    # Pass the current data dictionary
                                prior_steps_list,    # Pass the list of prior steps
                                analysis_plan,       # Pass the current plan (gets updated in loop)
                                filtered_metadata_dict # Pass the user-filtered metadata
                            )

                            # Append step details for display
                            prior_steps_list.append(step_description) # Keep track of textual steps
                            step_info_for_display = {"step": step_description, "tool": tool_called}
                            if tool_called == 'SQL Tool' and isinstance(output, dict):
                                step_info_for_display["sql"] = output.get('SQL_Query')
                            st.session_state.prior_steps_display.append(step_info_for_display)

                            # Indication of progress:
                            print(step_info_for_display) # Keep for debugging if needed

                            # Check if the loop should terminate
                            if tool_called == 'Talk with user Tool':
                                talk_to_user_flag = True
                                final_output = output # The output is the message string
                            # Update plan in session state if it was modified (optional, depending on how update_plan works)
                            # st.session_state.plan = analysis_plan # Consider if needed persistently

                        if step_counter >= max_steps:
                             final_output = "Analysis stopped after reaching the maximum step limit."
                             st.warning(final_output)


                    # Store results in session state after the loop finishes successfully or hits max steps
                    st.session_state.analysis_results = final_output
                    st.session_state.new_data_results = current_new_data


                except Exception as e:
                    # Error occurred during the loop execution
                    st.error(f"An error occurred during the analysis execution loop: {e}")
                    st.error(traceback.format_exc())
                    st.session_state.analysis_results = f"Analysis stopped due to error: {e}"
                    # Store partial results if any
                    st.session_state.new_data_results = current_new_data
                    # Don't rerun on error, let user see the error message

                # Rerun ONLY if analysis completed (or max steps hit) without runtime errors to display results
                # Use a flag or check if 'e' was defined in the try block
                analysis_error_occurred = 'e' in locals()
                if not analysis_error_occurred:
                    st.rerun()

            else:
                # --- Selection differs: Re-run Planning ---
                st.warning("Your table selection differs from the previous suggestion. Re-generating the plan based on your current selection...")

                try:
                    # Filter metadata based on the NEW user selection
                    filtered_metadata_dict_for_replanning = {table: metadata_dict[table] for table in selected_tables if table in metadata_dict}

                    if not filtered_metadata_dict_for_replanning:
                         st.error("Cannot re-plan: None of the selected tables were found in the metadata.")
                    else:
                         with st.spinner("Re-generating analysis plan..."):
                             # Call create_plan agent AGAIN with the filtered metadata
                             # Read the question directly from session state
                             new_plan_result, new_suggested_tables = create_plan(
                                 question=st.session_state.user_question_input, # Read from state
                                 metadata=filtered_metadata_dict_for_replanning # Use filtered metadata
                             )

                         # Update session state with the NEW plan and suggestions
                         st.session_state.plan = new_plan_result['Plan']
                         # Validate the NEW suggestions against all available tables (good practice)
                         valid_new_suggested = [t for t in new_suggested_tables if t in all_available_tables]
                         # IMPORTANT: Update suggested_tables to the newly generated ones.
                         # The multiselect's default will now reflect these new suggestions on rerun.
                         st.session_state.suggested_tables = valid_new_suggested

                         # Rerun to show the updated multiselect with new defaults
                         st.rerun()

                except Exception as e:
                    st.error(f"Error during re-planning phase: {e}")
                    st.error(traceback.format_exc())
                    # Keep planning_complete True so the user can try again? Or set to False?
                    # Let's keep it True so they stay in the confirmation loop.


# --- Results Display Section ---

# Display steps in the sidebar (reads from session state)
with st.sidebar:
    st.header("Analysis Steps")
    if not st.session_state.prior_steps_display:
        st.write("No analysis steps executed yet.")
    else:
        for i, item in enumerate(st.session_state.prior_steps_display):
            st.write(item.get("step", f"Step {i+1} details missing"))
            if item.get("tool") == 'SQL Tool' and item.get("sql"):
                st.code(item["sql"], language="sql")
            st.markdown("---") # Separator

# Display final message and dataframes in the main area if results exist
if st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Analysis Result")
    st.markdown(st.session_state.analysis_results) # Display the final message

    # Find and display dataframe previews and download buttons
    # Use the stored new_data_results from session state
    generated_data = st.session_state.new_data_results
    # Ensure analysis_results is a string before using regex
    results_text = str(st.session_state.analysis_results) if st.session_state.analysis_results else ""
    matches = re.findall(r'\[\[(.*?)\]\]', results_text) # Find [[df_name]] mentions
    displayed_dfs = set()

    if matches:
        st.subheader("Generated DataFrames")
        for name in matches:
            name = name.strip() # Clean whitespace
            if name in generated_data and name not in displayed_dfs:
                df_info = generated_data[name]
                # Assuming structure is {'df_name': {'data_frame': pd.DataFrame, ...}}
                df = df_info.get('data_frame')

                if isinstance(df, pd.DataFrame):
                    st.write(f"**DataFrame: `{name}`** (Preview)")
                    st.dataframe(df.head(), use_container_width=True)

                    csv_data = convert_df_to_csv(df) # Use the cached function
                    if csv_data:
                        try:
                            # Use a more robust unique key including potentially dataframe hash or length
                            df_hash = pd.util.hash_pandas_object(df).sum()
                            st.download_button(
                                label=f"Download `{name}` as CSV",
                                data=csv_data,
                                file_name=f"{name}.csv",
                                mime='text/csv',
                                key=f"download_{name}_{df_hash}" # Make key unique
                            )
                        except Exception as e:
                            st.error(f"Error creating download button for {name}: {e}")

                    displayed_dfs.add(name) # Mark as displayed
                else:
                      st.warning(f"Data for `[[{name}]]` found but is not a valid DataFrame.")
                      displayed_dfs.add(name) # Avoid re-warning

            elif name not in generated_data and name not in displayed_dfs:
                # Only warn if mentioned but not found AND not already warned about
                st.warning(f"DataFrame `[[{name}]]` mentioned in the result but was not found in the generated data.")
                displayed_dfs.add(name) # Avoid re-warning

# --- End of App ---
