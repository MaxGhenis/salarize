import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.figure_factory as ff
import plotly.graph_objects as go
from collections import defaultdict
import os
from anthropic import Anthropic
import re

# Assuming API key and client setup
api_key = os.getenv("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
client = Anthropic(api_key=api_key)


def generate_query_for_quantiles(job_title, company, location, model_type):
    inflation_note = (
        "Note: Prices have risen by 11% from January 2022 to March 2024."
    )
    return (
        f"{inflation_note} I need the specific numerical salary estimates only, no explanations, "
        f"for the {job_title} role at {company} in {location}. "
        f"Provide the data in a numerical list separated by commas for the following percentiles "
        f"10th, 25th, 50th, 75th, and 90th. For example: '10: <salary>, 25: <salary>, "
        f"50: <salary>, 75: <salary>, 90: <salary>'. Only include the salary numbers and percentiles."
    )


def extract_quantiles_from_response(text):
    # This regex matches the percentile indicator followed by a colon, optional space,
    # an optional dollar sign, and a number that may contain commas.
    # It captures the percentile and the corresponding number.
    quantile_pattern = re.compile(r"(\d+): \$?([\d,]+)")

    # Remove any periods that denote the end of a sentence, not decimal points
    clean_text = text.replace(".", "")

    # Find all matches and convert them to a dictionary
    # The first capture group is the percentile, the second is the salary.
    # We remove commas from the salary for proper conversion to int.
    quantiles = {
        int(match[0]): int(match[1].replace(",", ""))
        for match in quantile_pattern.findall(clean_text)
    }
    return quantiles


def synthesize_quantiles(quantiles_list):
    # Ensure that the quantile synthesis is done correctly
    synthesized_quantiles = defaultdict(list)
    for quantiles in quantiles_list:
        for q, value in quantiles.items():
            synthesized_quantiles[q].append(value)
    averaged_quantiles = {
        q: np.mean(values) for q, values in synthesized_quantiles.items()
    }
    return averaged_quantiles


def estimate_salary(query, model_type):
    # Implement the actual API call here
    # Example (this needs to be replaced with the real implementation):
    model_map = {
        "Haiku": "claude-3-haiku-20240307",
        "Sonnet": "claude-3-sonnet-20240229",
        "Opus": "claude-3-opus-20240229",
    }
    try:
        response = client.messages.create(
            max_tokens=50,
            model=model_map[model_type],
            messages=[{"role": "user", "content": query}],
        )
        if (
            response.content
            and isinstance(response.content, list)
            and hasattr(response.content[0], "text")
        ):
            return response.content[0].text
    except Exception as e:
        print("Error during API call:", e)
    return "Error or no data"


def fit_and_plot_distribution(averaged_quantiles):
    # Fit a distribution to the averaged quantiles
    quantile_values = list(averaged_quantiles.values())
    quantiles = np.array(quantile_values)
    log_quantiles = np.log(quantiles)
    shape, loc, scale = stats.lognorm.fit(log_quantiles, floc=0)

    # Generate a range of salary values based on the quantiles
    salary_range = np.linspace(quantiles.min(), quantiles.max(), 100)
    # Compute the PDF on the original scale
    pdf = stats.lognorm.pdf(salary_range, shape, scale=np.exp(scale), loc=loc)

    # Create the plot on the original scale
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=salary_range, y=pdf, mode="lines", name="Fitted PDF")
    )
    fig.update_layout(
        title="Predicted Salary Distribution",
        xaxis_title="Salary",
        yaxis_title="Density",
    )
    return fig


def main():
    st.title("Salarizer™")
    st.write("What does AI think your job is paid?")

    model_type = st.selectbox("Choose Model Type", ("Haiku", "Sonnet", "Opus"))
    job_title = st.text_input("Job Title")
    company = st.text_input("Company")
    location = st.text_input("Location")
    num_queries = st.number_input(
        "Number of Queries", min_value=1, max_value=100, value=10, step=1
    )

    if st.button("Estimate Salary Distribution"):
        query = generate_query_for_quantiles(
            job_title, company, location, model_type
        )
        quantiles_list = []
        for _ in range(num_queries):
            response_text = estimate_salary(query, model_type)
            print(response_text)
            if (
                "10:" in response_text
            ):  # Check if the response is in the expected format
                quantiles = extract_quantiles_from_response(response_text)
                quantiles_list.append(quantiles)
            else:
                st.error(f"Unexpected response format: {response_text}")

        if quantiles_list:
            try:
                synthesized_quantiles = synthesize_quantiles(quantiles_list)
                median_salary = synthesized_quantiles.get(
                    50, "No median found"
                )
                st.success(
                    f"Claude predicts a median salary of ${median_salary:,.0f}"
                )
                fig = fit_and_plot_distribution(synthesized_quantiles)
                st.plotly_chart(fig)

            except ValueError as e:
                st.error(str(e))
        else:
            st.error("Failed to retrieve valid quantile data.")


if __name__ == "__main__":
    main()
