import streamlit as st
import pandas as pd
from agents.orchestrator_agent import orchestrator_agent
from agents.sql_agent import sql_agent
from agents.talk_with_user_agent import talk_with_user_agent
from agents.update_plan import update_plan
from agents.create_plan import create_plan
from utils.unique import metadata_str_into_dict, create_filtered_dict, format_dict_info
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
            step_output = message.get('Send_To_User', 'No message generated.')
            step_details = f"'Message to User: {step_output}'"
            return tool_to_call, step_output, new_data, f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}", plan # Return user message directly

        elif tool_to_call == 'SQL Tool':
            # Ensure metadata for the specific DB is passed correctly
            db_metadata = filtered_dict.get(DB_NAME, {})
            if not db_metadata:
                 raise ValueError(f"No metadata found for database '{DB_NAME}' in the filtered dictionary.")

            ai_output, new_data = sql_agent(
                instructions,
                metadata = db_metadata,
                database_name = DB_NAME,
                new_data = new_data # Pass and potentially update new_data dict
            )
            df_name = ai_output.get('df_name', 'Unknown DataFrame')
            step_details = f"'{instructions}' | OUTPUT DataFrame: {df_name}"
            # Update the plan (optional, depends on update_plan logic)
            updated_plan = update_plan(prior_steps + [f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}"], plan)

            # Store SQL query for display if available
            sql_query = ai_output.get('SQL_Query', None)
            return tool_to_call, {'df_name': df_name, 'SQL_Query': sql_query}, new_data, f"STEP {len(prior_steps) + 1}. [{tool_to_call}]: {step_details}", updated_plan

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

# Main area for user input
question = st.text_area("Enter your question:", key="user_question_input")

# --- Planning Phase ---
col1, col2 = st.columns([1, 3]) # Adjust column widths as needed

with col1:
    if st.button("Plan Analysis", key="plan_button"):
        if question:
            # Reset state for a new planning action
            st.session_state.planning_complete = False
            st.session_state.analysis_results = None
            st.session_state.new_data_results = {}
            st.session_state.prior_steps_display = []

            try:
                with st.spinner("Generating analysis plan..."):
                    # Call create_plan agent
                    plan_result, suggested_tables = create_plan(question=question, metadata_string=metadata_string)
                st.session_state.plan = plan_result
                # Validate suggested tables against all available tables
                valid_suggested = [t for t in suggested_tables if t in all_available_tables]
                st.session_state.suggested_tables = valid_suggested
                st.session_state.planning_complete = True
                # No rerun needed here, the rest of the script will execute and show the next section

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
    confirmed_tables = st.multiselect(
        "Select tables to include in the analysis:",
        options=all_available_tables,
        default=st.session_state.suggested_tables, # Pre-select suggested tables
        key="table_confirmation_multiselect" # Key to access the selection later
    )

    # Add the "Run Analysis" button within this conditional block
    if st.button("Run Analysis with Selected Tables", key="run_analysis_button"):
        selected_tables = st.session_state.table_confirmation_multiselect # Get current selection
        if not selected_tables:
            st.warning("Please select at least one table to run the analysis.")
        else:
            # --- Analysis Execution ---
            st.info(f"Starting analysis using tables: `{'`, `'.join(selected_tables)}`")

            # Create filtered metadata based on the user's confirmed selection
            try:
                filtered_metadata_dict = create_filtered_dict(metadata_dict, selected_tables, db_name=DB_NAME)
            except Exception as e:
                 st.error(f"Failed to filter metadata based on selection: {e}")
                 st.stop()

            # Initialize/reset state for the analysis run
            prior_steps_list = []
            current_new_data = {} # Use a local variable for the loop
            analysis_plan = st.session_state.plan # Get the plan from session state
            talk_to_user_flag = False
            final_output = "Analysis could not be completed." # Default final message

            # Clear previous steps display
            st.session_state.prior_steps_display = []

            try:
                with st.spinner("Running analysis steps..."):
                    # The main analysis loop
                    while not talk_to_user_flag:
                        # Call the run_analysis function for one step
                        tool_called, output, current_new_data, step_description, analysis_plan = run_analysis(
                            question,           # Pass the original question
                            current_new_data,   # Pass the current data dictionary
                            prior_steps_list,   # Pass the list of prior steps
                            analysis_plan,      # Pass the current plan
                            filtered_metadata_dict # Pass the user-filtered metadata
                        )

                        # Append step details for display
                        prior_steps_list.append(step_description) # Keep track of textual steps
                        step_info_for_display = {"step": step_description, "tool": tool_called}
                        if tool_called == 'SQL Tool' and isinstance(output, dict):
                             step_info_for_display["sql"] = output.get('SQL_Query')
                        st.session_state.prior_steps_display.append(step_info_for_display)


                        # Check if the loop should terminate
                        if tool_called == 'Talk with user Tool':
                            talk_to_user_flag = True
                            final_output = output # The output is the message string

                # Store results in session state after the loop finishes successfully
                st.session_state.analysis_results = final_output
                st.session_state.new_data_results = current_new_data
                # Optionally update the plan if it was modified: st.session_state.plan = analysis_plan

            except Exception as e:
                # Error occurred during the loop execution
                st.error(f"An error occurred during the analysis execution loop: {e}")
                st.error(traceback.format_exc())
                st.session_state.analysis_results = f"Analysis stopped due to error: {e}"
                st.session_state.new_data_results = current_new_data # Store partial results if any


            # Rerun to display results sections now that they are populated
            st.rerun()


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
    matches = re.findall(r'\[\[(.*?)\]\]', st.session_state.analysis_results) # Find [[df_name]] mentions
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
                            st.download_button(
                                label=f"Download `{name}` as CSV",
                                data=csv_data,
                                file_name=f"{name}.csv",
                                mime='text/csv',
                                key=f"download_{name}_{len(csv_data)}" # Make key potentially unique based on data
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
