import { Chart } from "@/components/ui/chart"
document.addEventListener("DOMContentLoaded", () => {
  // Animate total predictions counter
  animateCounter()

  // Load recent predictions
  loadRecentPredictions()

  // Initialize feature importance chart
  initializeFeatureChart()

  function animateCounter() {
    const counter = document.getElementById("total-predictions")
    const target = 10247
    const duration = 2000
    const startTime = performance.now()

    function updateCounter(currentTime) {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const current = Math.floor(target * easeOutCubic(progress))

      counter.textContent = current.toLocaleString()

      if (progress < 1) {
        requestAnimationFrame(updateCounter)
      }
    }

    requestAnimationFrame(updateCounter)
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3)
  }

  function loadRecentPredictions() {
    const recentPredictions = [
      { id: 1, risk: "Low", probability: 23.5, timestamp: "2024-01-15 14:30" },
      { id: 2, risk: "High", probability: 78.2, timestamp: "2024-01-15 13:45" },
      { id: 3, risk: "Moderate", probability: 45.7, timestamp: "2024-01-15 12:20" },
      { id: 4, risk: "Low", probability: 18.9, timestamp: "2024-01-15 11:15" },
      { id: 5, risk: "High", probability: 82.1, timestamp: "2024-01-15 10:30" },
    ]

    const container = document.getElementById("recent-predictions")
    container.innerHTML = ""

    recentPredictions.forEach((prediction) => {
      const predictionItem = document.createElement("div")
      predictionItem.className = "prediction-item"
      predictionItem.innerHTML = `
                <div class="prediction-info">
                    <div class="prediction-badge ${prediction.risk.toLowerCase()}">${prediction.risk}</div>
                    <div class="prediction-details">
                        <div class="prediction-risk">${prediction.probability}% Risk</div>
                        <div class="prediction-time">${prediction.timestamp}</div>
                    </div>
                </div>
                <div class="prediction-status">
                    <i class="fas fa-check-circle"></i>
                </div>
            `
      container.appendChild(predictionItem)
    })
  }

  function initializeFeatureChart() {
    const ctx = document.getElementById("featureChart").getContext("2d")

    const featureImportance = [
      { feature: "Glucose Level", importance: 28 },
      { feature: "BMI", importance: 22 },
      { feature: "Age", importance: 18 },
      { feature: "Diabetes Pedigree", importance: 15 },
      { feature: "Blood Pressure", importance: 12 },
      { feature: "Insulin", importance: 8 },
      { feature: "Pregnancies", importance: 5 },
      { feature: "Skin Thickness", importance: 3 },
    ]

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: featureImportance.map((item) => item.feature),
        datasets: [
          {
            label: "Feature Importance (%)",
            data: featureImportance.map((item) => item.importance),
            backgroundColor: [
              "rgba(59, 130, 246, 0.8)",
              "rgba(139, 92, 246, 0.8)",
              "rgba(236, 72, 153, 0.8)",
              "rgba(34, 197, 94, 0.8)",
              "rgba(251, 146, 60, 0.8)",
              "rgba(168, 85, 247, 0.8)",
              "rgba(14, 165, 233, 0.8)",
              "rgba(99, 102, 241, 0.8)",
            ],
            borderColor: [
              "rgba(59, 130, 246, 1)",
              "rgba(139, 92, 246, 1)",
              "rgba(236, 72, 153, 1)",
              "rgba(34, 197, 94, 1)",
              "rgba(251, 146, 60, 1)",
              "rgba(168, 85, 247, 1)",
              "rgba(14, 165, 233, 1)",
              "rgba(99, 102, 241, 1)",
            ],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            titleColor: "white",
            bodyColor: "white",
            borderColor: "rgba(59, 130, 246, 1)",
            borderWidth: 1,
            cornerRadius: 8,
            callbacks: {
              label: (context) => `Importance: ${context.parsed.y}%`,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 30,
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
            ticks: {
              callback: (value) => value + "%",
            },
          },
          x: {
            grid: {
              display: false,
            },
            ticks: {
              maxRotation: 45,
              minRotation: 45,
            },
          },
        },
        animation: {
          duration: 2000,
          easing: "easeOutCubic",
        },
      },
    })
  }
})
