// Mobile Navigation Toggle
document.addEventListener("DOMContentLoaded", () => {
  const hamburger = document.getElementById("hamburger")
  const navMenu = document.getElementById("nav-menu")

  if (hamburger && navMenu) {
    hamburger.addEventListener("click", () => {
      navMenu.classList.toggle("active")
      hamburger.classList.toggle("active")
    })
  }

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })

  // Animate stats on scroll
  const observerOptions = {
    threshold: 0.5,
    rootMargin: "0px 0px -100px 0px",
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateStats()
        observer.unobserve(entry.target)
      }
    })
  }, observerOptions)

  const statsSection = document.querySelector(".hero-stats")
  if (statsSection) {
    observer.observe(statsSection)
  }

  function animateStats() {
    const statNumbers = document.querySelectorAll(".stat-number")
    statNumbers.forEach((stat) => {
      const text = stat.textContent
      if (text.includes("%")) {
        animateNumber(stat, 0, Number.parseFloat(text), "%")
      } else if (text.includes("K+")) {
        animateNumber(stat, 0, Number.parseFloat(text), "K+")
      } else {
        animateNumber(stat, 0, Number.parseInt(text), "")
      }
    })
  }

  function animateNumber(element, start, end, suffix) {
    const duration = 2000
    const startTime = performance.now()

    function update(currentTime) {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const current = start + (end - start) * easeOutCubic(progress)

      if (suffix === "%") {
        element.textContent = current.toFixed(1) + suffix
      } else if (suffix === "K+") {
        element.textContent = current.toFixed(1) + suffix
      } else {
        element.textContent = Math.floor(current) + suffix
      }

      if (progress < 1) {
        requestAnimationFrame(update)
      }
    }

    requestAnimationFrame(update)
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3)
  }
})

// Add loading states for buttons
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("btn-primary")) {
    const button = e.target
    const originalText = button.innerHTML

    // Don't add loading state for navigation buttons
    if (button.getAttribute("href")) {
      return
    }

    button.innerHTML = '<div class="loading"></div> Loading...'
    button.disabled = true

    setTimeout(() => {
      button.innerHTML = originalText
      button.disabled = false
    }, 2000)
  }
})
