document.getElementById("loanForm").addEventListener("submit", async function (event) {

    event.preventDefault();

    const data = {
        annual_income: Number(document.getElementById("annual_income").value),
        debt_to_income_ratio: Number(document.getElementById("debt_to_income_ratio").value),
        credit_score: Number(document.getElementById("credit_score").value),
        loan_amount: Number(document.getElementById("loan_amount").value),
        interest_rate: Number(document.getElementById("interest_rate").value),

        gender: document.getElementById("gender").value,
        marital_status: document.getElementById("marital_status").value,
        education_level: document.getElementById("education_level").value,
        employment_status: document.getElementById("employment_status").value,
        loan_purpose: document.getElementById("loan_purpose").value,
        grade_subgrade: document.getElementById("grade_subgrade").value
    };

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        console.log("API response:", result);

        if (!result.success) {
            alert("Prediction error: " + result.error);
            return;
        }

        const predictionResult = result.result;

        document.getElementById("result").classList.remove("hidden");

        document.getElementById("probability").textContent =
            (predictionResult.probability * 100).toFixed(2) + "%";

        document.getElementById("prediction").textContent =
            predictionResult.prediction === 1
                ? "Loan Paid Back"
                : "Loan Defaulted";

        document.getElementById("threshold").textContent =
            predictionResult.threshold_used;

    } catch (error) {

        console.error(error);

        alert("Could not connect to prediction API.");
    }

});