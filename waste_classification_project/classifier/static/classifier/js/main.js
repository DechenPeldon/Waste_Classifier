// Main JavaScript file for additional functionality
console.log('Smart Waste Classifier loaded successfully');

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});

// Add loading animation
document.addEventListener('DOMContentLoaded', function() {
    // Animate confidence bars on result page
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        // read numeric value from data attribute and animate
        const val = parseFloat(confidenceFill.dataset.conf || 0);
        const width = (isNaN(val) ? 0 : val) + '%';
        confidenceFill.style.width = '0';
        setTimeout(() => {
            confidenceFill.style.width = width;
        }, 100);
    }
    
    // Animate prediction bars
    const predBars = document.querySelectorAll('.pred-bar');
    predBars.forEach((bar, index) => {
        const val = parseFloat(bar.dataset.conf || 0);
        const width = (isNaN(val) ? 0 : val) + '%';
        bar.style.width = '0';
        setTimeout(() => {
            bar.style.width = width;
        }, 100 + (index * 50));
    });
});