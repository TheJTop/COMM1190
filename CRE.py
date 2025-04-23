import streamlit as st
import pandas as pd
from agents.orchestrator_agent import orchestrator_agent
from agents.sql_agent import sql_agent
from agents.talk_with_user_agent import talk_with_user_agent
from agents.update_plan import update_plan
from agents.create_plan import create_plan
from utils.unique import metadata_str_into_dict, create_filtered_dict, format_dict_info

import re

DB_NAME = 'CRE_Data'

# Initialize the Streamlit app
st.title("CRE Database Chat")

# Create a tab for the main page and the dataframes tab
tab_main = st.tabs(["Main"])[0]

# Initialize the metadata and new_data dictionaries
with open('agents/prompts/metadata.txt', 'r') as file:
    metadata = file.read()
    metadata_dict = metadata_str_into_dict(metadata)

#print(metadata_dict['CRE_Data.db'].keys()) #How to get list of tables
new_data = {}
prior_steps = []
plan = ""

# Create a function to handle the user input and run the analysis
def run_analysis(question, new_data, prior_steps, plan, filtered_dict):

    # Call the orchestrator_agent
    output = orchestrator_agent(question=question, metadata=filtered_dict, new_data=new_data, prior_steps=prior_steps, current_plan=plan)
    
    # Call the next agent
    if output['Call_Tool']['Tool'] == 'Talk with user Tool':
        message = talk_with_user_agent(
            user_message=output['Call_Tool']['Instructions'],
            metadata=filtered_dict,
            new_data=new_data,
            prior_steps=prior_steps,
            current_plan=plan
            )
        
        step = f"STEP {len(prior_steps) + 1}. [{output['Call_Tool']['Tool']}]: 'Message to User: {message['Send_To_User']}"
    
    elif output['Call_Tool']['Tool'] == 'SQL Tool':
        ai_output, new_data = sql_agent(
            output['Call_Tool']['Instructions'],
            metadata = filtered_dict['CRE_Data.db'],
            database_name = 'CRE_Data',
            new_data = new_data
            )
        step = f"STEP {len(prior_steps) + 1}. [{output['Call_Tool']['Tool']}]: '{output['Call_Tool']['Instructions']}' | OUTPUT: {ai_output['df_name']}"
    
    prior_steps.append(step)

    # Update the plan
    plan = update_plan(prior_steps, plan)

    if output['Call_Tool']['Tool'] == 'Talk with user Tool':
        return output['Call_Tool']['Tool'], message['Send_To_User'], new_data, step, plan
    elif output['Call_Tool']['Tool'] == 'SQL Tool':
        return output['Call_Tool']['Tool'], ai_output, new_data, step, plan

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False)
# Create the main page
with tab_main:
    question = st.text_area("Enter your question")
    if st.button("Run"):
        talk_with_user = False
        # Create the initial plan
        plan, tables = create_plan(question = question, metadata_string = metadata)
        
        # Get tables from the plan to limit metadata
        filtered_dict = create_filtered_dict(metadata_dict, tables)
        filtered_metadata = format_dict_info(filtered_dict)



        while talk_with_user == False:
            tool_called, output, new_data, step, plan  = run_analysis(question, new_data, prior_steps, plan, filtered_dict)
            with st.sidebar:
                st.write(step)
                if tool_called == 'SQL Tool':
                    st.write(f"{output['SQL_Query']}")
            if tool_called == 'Talk with user Tool':
                talk_with_user = True
                st.write(output)

        matches = re.findall(r'\[\[(.*?)\]\]', output)

        for name in matches:
            st.write(f"Dataframe: {name}")
            st.write(new_data[name]['data_frame'].head())
            
            csv = convert_df(new_data[name]['data_frame'])
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{name}.csv",
                mime='text/csv',
            )
