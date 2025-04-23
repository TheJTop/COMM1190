import streamlit as st
import pandas as pd
from agents.orchestrator_agent import orchestrator_agent
from agents.sql_agent import sql_agent
from agents.talk_with_user_agent import talk_with_user_agent
from agents.update_plan import update_plan
from agents.create_plan import create_plan
from utils.unique import metadata_str_into_dict, create_filtered_dict, format_dict_info # Assuming these utils exist and work as expected
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
if 'filtered_metadata' not in st.session_state:
    st.session_state.filtered_metadata = None # Store filtered metadata for analysis steps
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False # Flag to control step-by-step execution
if 'current_question' not in st.session_state:
    st.session_state.current_question = "" # Store the question being analyzed

# --- Load Metadata ---
# Load once and store potentially in session state or rely on Streamlit's script execution model
try:
    with open('agents/prompts/metadata.txt', 'r') as file:
         metadata_string = file.read() # Keep the raw string for create_plan
         metadata_dict = metadata_str_into_dict(metadata_string)
         # Get all available tables for the database
         all_available_tables = list(metadata_dict.get(DB_NAME, {}).keys())
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

def run_single_analysis_step(question, new_data, prior_steps, plan, filtered_dict):
    """Runs ONLY ONE step of the analysis using the orchestrator and appropriate tool."""
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
            step_output = message.get('Send_To_User', 'No message generated.')
            step_details = f"'Message to User: {step_output}'"
            step_description = f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}"
            step_info_for_display = {"step": step_description, "tool": tool_to_call}
            return tool_to_call, step_output, new_data, step_info_for_display, plan # Return user message directly

        elif tool_to_call == 'SQL Tool':
            # Ensure metadata for the specific DB is passed correctly
            db_metadata = filtered_dict.get(DB_NAME, {})
            if not db_metadata:
                 raise ValueError(f"No metadata found for database '{DB_NAME}' in the filtered dictionary.")

            ai_output, updated_new_data = sql_agent(
                 instructions,
                 metadata = db_metadata,
                 database_name = DB_NAME,
                 new_data = new_data # Pass and potentially update new_data dict
            )
            df_name = ai_output.get('df_name', 'Unknown DataFrame')
            step_details = f"'{instructions}' | OUTPUT DataFrame: {df_name}"
            step_description = f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}"

            # Update the plan (optional, depends on update_plan logic)
            # Ensure update_plan can handle being called repeatedly if needed.
            # For simplicity here, we might assume the initial plan is sufficient or handle updates carefully.
            # Let's assume update_plan is only called if needed based on some logic, or maybe not needed per step.
            # For this example, we pass the plan through but don't explicitly update it here unless sql_agent's output suggests it.
            # updated_plan = update_plan(prior_steps + [step_description], plan) # If plan needs updating based on step
            updated_plan = plan # Keep plan as is for now unless specifically needed

            # Store SQL query for display if available
            sql_query = ai_output.get('SQL_Query', None)
            step_info_for_display = {"step": step_description, "tool": tool_to_call, "sql": sql_query}
            return tool_to_call, {'df_name': df_name, 'SQL_Query': sql_query}, updated_new_data, step_info_for_display, updated_plan

        else:
             # Handle unknown tool case gracefully
             st.warning(f"Orchestrator requested an unknown tool: {tool_to_call}")
             step_output = f"Attempted to use unknown tool: {tool_to_call}"
             step_description = f"STEP {len(prior_steps) + 1}. [Unknown Tool]: {tool_to_call}"
             step_info_for_display = {"step": step_description, "tool": "Unknown Tool"}
             # Decide how to proceed: maybe treat as 'Talk with user' or stop?
             # For now, treat as end step with a warning message.
             return 'Talk with user Tool', f"Analysis stopped: Unknown tool '{tool_to_call}' requested.", new_data, step_info_for_display, plan


    except Exception as e:
        st.error(f"Error during analysis step execution: {e}")
        st.error(traceback.format_exc())
        # Return an error indicator to stop the process gracefully
        error_message = f"Analysis stopped due to error: {e}"
        step_description = f"STEP {len(prior_steps) + 1}. [ERROR]: {e}"
        step_info_for_display = {"step": step_description, "tool": "Error"}
        # Treat error as an end condition ('Talk with user Tool' signals end)
        return 'Talk with user Tool', error_message, new_data, step_info_for_display, plan

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

# Main area for user input
# Use session state to preserve the question if analysis is in progress
if 'user_question_input' not in st.session_state:
     st.session_state.user_question_input = ""

question = st.text_area("Enter your question:", key="user_question_input")

# --- Planning Phase ---
col1, col2 = st.columns([1, 3]) # Adjust column widths as needed

with col1:
    # Disable planning if analysis is running
    plan_button_disabled = st.session_state.analysis_in_progress
    if st.button("Plan Analysis", key="plan_button", disabled=plan_button_disabled):
        if question:
            # Reset state for a new planning action
            st.session_state.planning_complete = False
            st.session_state.analysis_results = None
            st.session_state.new_data_results = {}
            st.session_state.prior_steps_display = []
            st.session_state.analysis_in_progress = False # Ensure analysis stops if a new plan is made
            st.session_state.filtered_metadata = None
            st.session_state.current_question = question # Store the question associated with this plan

            try:
                with st.spinner("Generating analysis plan..."):
                    # Call create_plan agent
                    plan_result, suggested_tables = create_plan(question=question, metadata_string=metadata_string)
                st.session_state.plan = plan_result
                # Validate suggested tables against all available tables
                valid_suggested = [t for t in suggested_tables if t in all_available_tables]
                st.session_state.suggested_tables = valid_suggested
                st.session_state.planning_complete = True
                st.rerun() # Rerun to show the table selection section immediately

            except Exception as e:
                st.error(f"Error during planning phase: {e}")
                st.error(traceback.format_exc())
                st.session_state.planning_complete = False # Ensure flag is false on error
                st.session_state.analysis_in_progress = False
        else:
            st.warning("Please enter a question before planning.")

# --- Table Confirmation and Execution Phase ---
# This section appears only after the 'Plan Analysis' button has been successfully clicked
if st.session_state.planning_complete:
    st.markdown("---") # Visual separator
    st.subheader("Table Selection Confirmation")
    st.markdown("The analysis plan suggests using the tables below. Please confirm or adjust the selection for the analysis:")

    # The multiselect widget manages its own state via the key
    confirmed_tables = st.multiselect(
        "Select tables to include in the analysis:",
        options=all_available_tables,
        default=st.session_state.suggested_tables, # Pre-select suggested tables
        key="table_confirmation_multiselect",
        disabled=st.session_state.analysis_in_progress # Disable if analysis is running
    )

    # Determine if the button should be "Run Analysis" or "Continue Analysis"
    run_button_label = "Continue Analysis" if st.session_state.analysis_in_progress else "Run Analysis with Selected Tables"
    run_button_key = "run_analysis_button"

    if st.button(run_button_label, key=run_button_key):
        selected_tables = st.session_state.table_confirmation_multiselect # Get current selection
        if not selected_tables:
            st.warning("Please select at least one table to run the analysis.")
        else:
            # --- Analysis Step Execution ---
            if not st.session_state.analysis_in_progress:
                # --- This is the FIRST step ---
                st.info(f"Starting analysis using tables: `{'`, `'.join(selected_tables)}`")
                # Create and store filtered metadata based on the user's confirmed selection
                try:
                    filtered_metadata_dict = create_filtered_dict(metadata_dict, selected_tables, db_name=DB_NAME)
                    st.session_state.filtered_metadata = filtered_metadata_dict # Store for subsequent steps
                except Exception as e:
                     st.error(f"Failed to filter metadata based on selection: {e}")
                     st.stop()

                # Initialize/reset state specifically for this analysis run
                st.session_state.prior_steps_display = []
                st.session_state.new_data_results = {}
                st.session_state.analysis_results = None # Clear previous final results
                # analysis_plan is already in st.session_state.plan
                st.session_state.analysis_in_progress = True # Set flag to start the process
                # Use the question stored when planning started
                current_question = st.session_state.current_question

            # --- Execute ONE analysis step (whether first or subsequent) ---
            # Retrieve necessary state from session state
            prior_steps_list_text = [item['step'] for item in st.session_state.prior_steps_display] # Get text desc only
            current_new_data = st.session_state.new_data_results
            analysis_plan = st.session_state.plan
            filtered_metadata_dict = st.session_state.filtered_metadata
            current_question = st.session_state.current_question # Get the question


            if not filtered_metadata_dict:
                 st.error("Filtered metadata is missing. Cannot proceed with analysis step.")
                 st.session_state.analysis_in_progress = False # Stop the process
                 st.rerun()

            else:
                try:
                    with st.spinner("Running next analysis step..."):
                        # Call the function to run just one step
                        tool_called, output, updated_new_data, step_info_for_display, updated_plan = run_single_analysis_step(
                            current_question,         # Pass the original question
                            current_new_data,       # Pass the current data dictionary
                            prior_steps_list_text,  # Pass the list of prior step descriptions
                            analysis_plan,          # Pass the current plan
                            filtered_metadata_dict  # Pass the user-filtered metadata
                        )

                    # Update session state with results of the step
                    st.session_state.prior_steps_display.append(step_info_for_display)
                    st.session_state.new_data_results = updated_new_data
                    st.session_state.plan = updated_plan # Update plan if modified

                    # Check if the step indicated the end of the analysis
                    if tool_called == 'Talk with user Tool':
                        st.session_state.analysis_results = output # Store the final message
                        st.session_state.analysis_in_progress = False # Mark analysis as complete
                    else:
                        # Analysis is not finished, keep the flag True
                        st.session_state.analysis_in_progress = True

                    # Rerun to display the updated sidebar and potentially continue
                    st.rerun()

                except Exception as e:
                    # Error occurred during the step execution within the function
                    st.error(f"A critical error occurred during analysis step execution: {e}")
                    st.error(traceback.format_exc())
                    st.session_state.analysis_results = f"Analysis stopped due to critical error: {e}"
                    st.session_state.analysis_in_progress = False # Stop the process
                    st.rerun() # Rerun to show the error state

# Add spinner indication if analysis is running but button wasn't just clicked
if st.session_state.analysis_in_progress and not st.session_state.get(run_button_key, False):
     st.info("Analysis in progress...")


# --- Results Display Section ---

# Display steps in the sidebar (reads from session state)
with st.sidebar:
    st.header("Analysis Steps")
    if not st.session_state.prior_steps_display:
        st.write("No analysis steps executed yet.")
    else:
        for i, item in enumerate(st.session_state.prior_steps_display):
            st.write(item.get("step", f"Step {i+1} details missing"))
            # Display SQL if available for the step
            if item.get("tool") == 'SQL Tool' and item.get("sql"):
                with st.expander("Show SQL Query", expanded=False):
                     st.code(item["sql"], language="sql")
            st.markdown("---") # Separator

# Display final message and dataframes in the main area if results exist
# This will only show when analysis_in_progress is False and analysis_results is set
if not st.session_state.analysis_in_progress and st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Analysis Result")
    st.markdown(st.session_state.analysis_results) # Display the final message

    # Find and display dataframe previews and download buttons
    # Use the stored new_data_results from session state
    generated_data = st.session_state.new_data_results
    # Check if analysis_results is a string before using regex
    final_message_str = st.session_state.analysis_results if isinstance(st.session_state.analysis_results, str) else ""
    matches = re.findall(r'\[\[(.*?)\]\]', final_message_str) # Find [[df_name]] mentions
    displayed_dfs = set()

    if matches or generated_data: # Also show DFs if they exist even if not mentioned in final text
        st.subheader("Generated DataFrames")

        # Display mentioned DFs first
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
                            st.download_button(
                                label=f"Download `{name}` as CSV",
                                data=csv_data,
                                file_name=f"{name}.csv",
                                mime='text/csv',
                                key=f"download_{name}_{len(csv_data)}" # Make key potentially unique
                            )
                        except Exception as e:
                            st.error(f"Error creating download button for {name}: {e}")

                    displayed_dfs.add(name) # Mark as displayed
                elif name not in displayed_dfs: # Only warn once if mentioned but not DF
                     st.warning(f"Data for `[[{name}]]` found but is not a valid DataFrame.")
                     displayed_dfs.add(name) # Avoid re-warning

            elif name not in generated_data and name not in displayed_dfs:
                # Only warn if mentioned but not found AND not already warned about
                st.warning(f"DataFrame `[[{name}]]` mentioned in the result but was not found in the generated data.")
                displayed_dfs.add(name) # Avoid re-warning

        # Display any other generated DFs that weren't mentioned
        for name, df_info in generated_data.items():
             if name not in displayed_dfs:
                 df = df_info.get('data_frame')
                 if isinstance(df, pd.DataFrame):
                    st.write(f"**DataFrame: `{name}`** (Preview - not explicitly mentioned in final result)")
                    st.dataframe(df.head(), use_container_width=True)
                    csv_data = convert_df_to_csv(df)
                    if csv_data:
                        try:
                            st.download_button(
                                label=f"Download `{name}` as CSV",
                                data=csv_data,
                                file_name=f"{name}.csv",
                                mime='text/csv',
                                key=f"download_{name}_{len(csv_data)}"
                            )
                        except Exception as e:
                            st.error(f"Error creating download button for {name}: {e}")
                    displayed_dfs.add(name)


# --- End of App ---
