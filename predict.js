document.addEventListener("DOMContentLoaded", () => {
  // Initialize slider value displays
  const sliders = document.querySelectorAll(".slider")
  sliders.forEach((slider) => {
    const valueDisplay = document.getElementById(slider.id + "-value")
    if (valueDisplay) {
      slider.addEventListener("input", function () {
        if (slider.id === "bmi" || slider.id === "diabetesPedigree") {
          valueDisplay.textContent = Number.parseFloat(this.value).toFixed(2)
        } else {
          valueDisplay.textContent = this.value
        }
      })
    }
  })

  // Form submission
  const form = document.getElementById("prediction-form")
  const resultsPlaceholder = document.getElementById("results-placeholder")
  const predictionResults = document.getElementById("prediction-results")
  const predictBtn = document.getElementById("predict-btn")

  form.addEventListener("submit", async (e) => {
    e.preventDefault()

    // Show loading state
    const originalBtnText = predictBtn.innerHTML
    predictBtn.innerHTML = '<div class="loading"></div> Analyzing...'
    predictBtn.disabled = true

    // Collect form data
    const formData = new FormData(form)
    const data = {
      pregnancies: Number.parseInt(formData.get("pregnancies")),
      glucose: Number.parseInt(formData.get("glucose")),
      bloodPressure: Number.parseInt(formData.get("bloodPressure")),
      skinThickness: Number.parseInt(formData.get("skinThickness")),
      insulin: Number.parseInt(formData.get("insulin")),
      bmi: Number.parseFloat(formData.get("bmi")),
      diabetesPedigree: Number.parseFloat(formData.get("diabetesPedigree")),
      age: Number.parseInt(formData.get("age")),
    }

    // Simulate API call delay
    await new Promise((resolve) => setTimeout(resolve, 1500))

    // Calculate prediction (simplified model)
    const prediction = calculateDiabetesRisk(data)

    // Display results
    displayResults(prediction)

    // Reset button
    predictBtn.innerHTML = originalBtnText
    predictBtn.disabled = false
  })

  function calculateDiabetesRisk(data) {
    let score = 0
    const factors = []
    const recommendations = []

    // Age factor
    if (data.age > 45) {
      score += 0.2
      factors.push("Advanced age (>45 years)")
    } else if (data.age > 35) {
      score += 0.1
      factors.push("Age factor (>35 years)")
    }

    // BMI factor
    if (data.bmi > 30) {
      score += 0.25
      factors.push("Obesity (BMI > 30)")
      recommendations.push("Consider weight management through diet and exercise")
    } else if (data.bmi > 25) {
      score += 0.15
      factors.push("Overweight (BMI > 25)")
      recommendations.push("Maintain healthy weight through balanced diet")
    }

    // Glucose factor
    if (data.glucose > 140) {
      score += 0.3
      factors.push("High glucose levels (>140 mg/dL)")
      recommendations.push("Monitor blood sugar levels regularly and consult a doctor")
    } else if (data.glucose > 120) {
      score += 0.2
      factors.push("Elevated glucose levels (>120 mg/dL)")
      recommendations.push("Monitor blood sugar levels and maintain healthy diet")
    }

    // Blood pressure factor
    if (data.bloodPressure > 90) {
      score += 0.15
      factors.push("High blood pressure (>90 mmHg)")
      recommendations.push("Monitor blood pressure and consider lifestyle changes")
    } else if (data.bloodPressure > 80) {
      score += 0.1
      factors.push("Elevated blood pressure (>80 mmHg)")
      recommendations.push("Maintain healthy blood pressure through exercise")
    }

    // Insulin factor
    if (data.insulin > 120) {
      score += 0.1
      factors.push("High insulin levels")
      recommendations.push("Consult healthcare provider about insulin levels")
    }

    // Diabetes pedigree factor
    if (data.diabetesPedigree > 0.5) {
      score += data.diabetesPedigree * 0.2
      factors.push("Family history of diabetes")
      recommendations.push("Regular screening due to family history")
    }

    // Pregnancies factor
    if (data.pregnancies > 3) {
      score += 0.1
      factors.push("Multiple pregnancies")
      recommendations.push("Regular health monitoring recommended")
    }

    // Skin thickness factor
    if (data.skinThickness > 30) {
      score += 0.05
      factors.push("Increased skin thickness")
    }

    // Convert to probability (0-100%)
    const probability = Math.min(Math.max(score * 100, 5), 95)

    let riskLevel
    if (probability < 30) {
      riskLevel = "Low"
    } else if (probability < 60) {
      riskLevel = "Moderate"
    } else {
      riskLevel = "High"
    }

    // Add general recommendations if no specific factors
    if (factors.length === 0) {
      factors.push("No significant risk factors identified")
      recommendations.push("Maintain current healthy lifestyle")
      recommendations.push("Continue regular health checkups")
    }

    // Add general healthy lifestyle recommendations
    if (recommendations.length < 3) {
      recommendations.push("Maintain a balanced diet rich in vegetables and whole grains")
      recommendations.push("Engage in regular physical activity (150 minutes per week)")
      recommendations.push("Schedule regular health screenings")
    }

    return {
      probability: probability,
      riskLevel: riskLevel,
      factors: factors,
      recommendations: recommendations,
    }
  }

  function displayResults(prediction) {
    // Hide placeholder and show results
    resultsPlaceholder.style.display = "none"
    predictionResults.style.display = "block"

    // Update risk percentage
    document.getElementById("risk-percentage").textContent = prediction.probability.toFixed(1) + "%"

    // Update risk badge
    const riskBadge = document.getElementById("risk-badge")
    riskBadge.textContent = prediction.riskLevel + " Risk"
    riskBadge.className = "risk-badge " + prediction.riskLevel.toLowerCase()

    // Update progress bar
    document.getElementById("progress-value").textContent = prediction.probability.toFixed(1) + "%"
    const progressFill = document.getElementById("progress-fill")
    progressFill.style.width = prediction.probability + "%"

    // Update risk factors
    const factorsList = document.getElementById("risk-factors-list")
    factorsList.innerHTML = ""
    prediction.factors.forEach((factor) => {
      const factorItem = document.createElement("div")
      factorItem.className = "factor-item"
      factorItem.innerHTML = `
                <div class="factor-dot"></div>
                <span>${factor}</span>
            `
      factorsList.appendChild(factorItem)
    })

    // Update recommendations
    const recommendationsList = document.getElementById("recommendations-list")
    recommendationsList.innerHTML = ""
    prediction.recommendations.forEach((recommendation) => {
      const recommendationItem = document.createElement("div")
      recommendationItem.className = "recommendation-item"
      recommendationItem.innerHTML = `
                <div class="recommendation-dot"></div>
                <span>${recommendation}</span>
            `
      recommendationsList.appendChild(recommendationItem)
    })

    // Animate progress bar
    setTimeout(() => {
      progressFill.style.transition = "width 1s ease-out"
    }, 100)

    // Scroll to results
    predictionResults.scrollIntoView({ behavior: "smooth", block: "nearest" })
  }
})
