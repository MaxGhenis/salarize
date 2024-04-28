import streamlit as st
import os
from anthropic import Anthropic
import re
import numpy as np
import plotly.express as px

# Check if API key is in the environment, otherwise pull from Streamlit secrets
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

api_key = os.environ["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=api_key)


def generate_query(job_title, company, location, model_type):
    inflation_note = (
        "Note: Prices have risen by 11% from January 2022 to March 2024."
    )
    return f"{inflation_note} Provide only the numerical estimated annual salary in USD for a {job_title} at {company} in {location}, without any additional text or explanation. Use data appropriate for a model type: {model_type}."


def extract_salary_from_response(text):
    # Adjusted regex to capture numbers with or without comma formatting
    salary_matches = re.findall(r"\$?\b\d+[\d,]*(?:\.\d+)?\b", text)
    if salary_matches:
        # Return only the first match assuming you expect one number
        return int(salary_matches[0].replace(",", "").replace("$", ""))
    return None


def estimate_salary(query, model_type):
    try:
        model_map = {
            "Haiku": "claude-3-haiku-20240307",
            "Sonnet": "claude-3-sonnet-20240229",
            "Opus": "claude-3-opus-20240229",
        }
        messages = [{"role": "user", "content": query}]
        response = client.messages.create(
            max_tokens=50,
            model=model_map[model_type],
            messages=messages,
        )
        if (
            response.content
            and isinstance(response.content, list)
            and hasattr(response.content[0], "text")
        ):
            return extract_salary_from_response(response.content[0].text)
        return None
    except Exception as e:
        print("Error during API call:", e)
        return None


def main():
    st.title("Salarize™")
    st.write("What does Claude AI think you should be paid?")

    model_type = st.selectbox("Choose Model Type", ("Haiku", "Sonnet", "Opus"))
    job_title = st.text_input("Job Title")
    company = st.text_input("Company")
    location = st.text_input("Location")
    num_queries = st.number_input(
        "Number of Queries", min_value=1, max_value=100, value=20, step=1
    )

    if st.button("Estimate Salary"):
        queries = [
            generate_query(job_title, company, location, model_type)
            for _ in range(num_queries)
        ]
        results = [estimate_salary(query, model_type) for query in queries]
        results = [result for result in results if result is not None]

        if results:
            median_salary = np.median(results)
            formatted_median = "${:,.0f}".format(median_salary)
            st.success(f"Median Estimated Salary: {formatted_median}")

            # Plotting the histogram using Plotly
            fig = px.histogram(
                results,
                nbins=10,
                labels={"value": "Salaries"},
                title="Distribution of Estimated Salaries",
            )
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig)
        else:
            st.error("Could not retrieve valid salary estimates.")


if __name__ == "__main__":
    main()
