import streamlit as st
import pandas as pd
from utils.core_utils import call_groq_api
from modules.sobol_sensitivity_analysis import get_sobol_context_for_chat
from modules.fast_analysis import get_fast_context_for_chat
from modules.ancova_analysis import get_ancova_context_for_chat
from modules.taylor_analysis import get_taylor_context_for_chat
from modules.ml_analysis import get_ml_context_for_chat
from modules.expectation_convergence_analysis import get_expectation_convergence_context_for_chat
from modules.exploratory_data_analysis import get_eda_context_for_chat

def sidebar_global_context_generator(prompt, current_code, selected_language_model):
    # If analyses have been run, include the analysis summary
    analysis_summary = ", ".join(list(st.session_state.all_results.keys()))
    
    # Start building a comprehensive context
    context = f"""
    You are an expert assistant helping users understand a comprehensive uncertainty quantification and sensitivity analysis report.
    
    The report includes results from the following analyses: {analysis_summary}
    
    The model being analyzed is defined as:
    ```python
    {current_code}
    ```
    """
    
    # Add basic information about each analysis if available
    if 'all_results' in st.session_state:
        context += "\n\nThe following analyses have been performed:\n\n"
        
        # Add Sobol results summary if available and relevant
        if "Sobol Analysis" in st.session_state.all_results:
            sobol_results = st.session_state.all_results["Sobol Analysis"]
            context += get_sobol_context_for_chat(sobol_results)
    
        # Add FAST results summary if available and relevant
        if "FAST Analysis" in st.session_state.all_results:
            fast_results = st.session_state.all_results["FAST Analysis"]
            context += get_fast_context_for_chat(fast_results)

        # Add ANCOVA results summary if available and relevant
        if "ANCOVA Analysis" in st.session_state.all_results:
            ancova_results = st.session_state.all_results["ANCOVA Analysis"]
            context += get_ancova_context_for_chat(ancova_results)

        # Add Taylor results summary if available and relevant
        if "Taylor Analysis" in st.session_state.all_results:
            taylor_results = st.session_state.all_results["Taylor Analysis"]
            context += get_taylor_context_for_chat(taylor_results)

        # Add Correlation Analysis results summary if available and relevant
        if "Correlation Analysis" in st.session_state.all_results:
            corr_results = st.session_state.all_results["Correlation Analysis"]
            all_corr = corr_results.get("all_correlation_results")
            if all_corr is not None:
                for output_name, corr_df in all_corr.items():
                    context += f"\n\n### Correlation Analysis Results for {output_name}\n"
                    context += corr_df.to_markdown()

        # Add HSIC Analysis results summary if available and relevant
        if "HSIC Analysis" in st.session_state.all_results:
            hsic_results = st.session_state.all_results["HSIC Analysis"]
            hsic_df = hsic_results.get("hsic_df")
            if hsic_df is not None:
                context += "\n\n### HSIC Sensitivity Analysis Results\n"
                context += hsic_df.to_markdown(index=False)

        # Add ML Analysis results summary if available and relevant
        if "ML Analysis" in st.session_state.all_results:
            ml_results = st.session_state.all_results["ML Analysis"]
            context += get_ml_context_for_chat(ml_results)

        # Add Expectation Convergence Analysis results summary if available and relevant
        if "Expectation Convergence Analysis" in st.session_state.all_results:
            exp_results = st.session_state.all_results["Expectation Convergence Analysis"]
            context += get_expectation_convergence_context_for_chat(exp_results)
    
        # Add Exploratory Data Analysis results summary if available and relevant
        if "Exploratory Data Analysis" in st.session_state.all_results:
            eda_results = st.session_state.all_results["Exploratory Data Analysis"]
            context += get_eda_context_for_chat(eda_results)
    
    # Add the user's question to the context
    context += f"\nPlease answer the user's question in the context of the full report. If necessary, refer to specific analyses by name.\n\nUser question: {prompt}"
    
    return context

def render_sidebar_chat(current_code, selected_language_model):
    st.sidebar.header("Chat about Results")
    if st.session_state.analyses_ran and 'all_results' in st.session_state:
        st.sidebar.info("Ask questions about your analysis results here.")
        if "sidebar_global_chat_messages" not in st.session_state:
            st.session_state.sidebar_global_chat_messages = []
        # Display existing chat messages in the sidebar
        for message in st.session_state.sidebar_global_chat_messages:
            with st.sidebar.chat_message(message["role"]):
                st.sidebar.write(message["content"])
        # Get user input in the sidebar
        sidebar_prompt = st.sidebar.chat_input("Ask about your analysis results...", key="sidebar_chat_input")
        if sidebar_prompt:
            st.session_state.sidebar_global_chat_messages.append({"role": "user", "content": sidebar_prompt})
            context = sidebar_global_context_generator(sidebar_prompt, current_code, selected_language_model)
            chat_history = ""
            if len(st.session_state.sidebar_global_chat_messages) > 1:
                chat_history = "Previous conversation:\n"
                for i, msg in enumerate(st.session_state.sidebar_global_chat_messages[:-1]):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n\n"
            chat_prompt = f"""
            {context}
            
            {chat_history}
            
            Current user question: {sidebar_prompt}
            
            Please provide a helpful, accurate response to this question.
            """
            with st.spinner("Thinking..."):
                try:
                    response_text = call_groq_api(chat_prompt, model_name=selected_language_model)
                except Exception as e:
                    st.sidebar.error(f"Error calling API: {str(e)}")
                    response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
            st.session_state.sidebar_global_chat_messages.append({"role": "assistant", "content": response_text})
            st.rerun()
    else:
        st.sidebar.warning("Chat will be available after you run UQ.")
