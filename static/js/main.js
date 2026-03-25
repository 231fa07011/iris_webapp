
document.addEventListener('DOMContentLoaded', () => {
    // 🌸 1. Progress Bar Animation
    const bars = document.querySelectorAll('.prob-bar-fill');
    
    // Function to trigger progress bar animations
    const animateBars = () => {
        bars.forEach(bar => {
            const finalWidth = bar.getAttribute('data-percentage');
            // Small delay for each bar
            setTimeout(() => {
                bar.style.width = finalWidth + '%';
            }, 300);
        });
    };

    // --- 2. Input Visual Feedback ---
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        // Update styling on value change
        input.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            const min = parseFloat(e.target.min);
            const max = parseFloat(e.target.max);
            
            if (val < min || val > max) {
                input.style.borderColor = 'rgba(239, 68, 68, 0.5)'; // Slate 500
                input.style.boxShadow = '0 0 0 4px rgba(239, 68, 68, 0.1)';
            } else {
                input.style.borderColor = '#10B981'; // Emerald 500
                input.style.boxShadow = '0 0 0 4px rgba(16, 185, 129, 0.1)';
            }
        });

        // Add smooth enter for inputs on page load
        input.style.transitionDelay = `${Math.random() * 0.5}s`;
    });

    // --- 3. Prediction Result Entrance ---
    const resultCard = document.querySelector('.results-section');
    if (resultCard) {
        // Fade in effect
        resultCard.style.opacity = '0';
        resultCard.style.transform = 'translateY(20px)';
        
        requestAnimationFrame(() => {
            resultCard.style.transition = 'all 0.8s cubic-bezier(0.16, 1, 0.3, 1)';
            resultCard.style.opacity = '1';
            resultCard.style.transform = 'translateY(0)';
            
            // Trigger bars after fade in
            animateBars();
        });
    }

    // --- 4. Predictive Visuals (Removed Custom Cursor) ---
    
    // --- 5. UI State Reset ---
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('reset', () => {
            const resultsSection = document.querySelector('.results-section');
            if (resultsSection) {
                // Return to empty state with fade out/in
                resultsSection.style.opacity = '0';
                setTimeout(() => {
                    resultsSection.innerHTML = `
                        <div class="empty-results">
                            <div class="pulsing-icon">🌸</div>
                            <h3 style="margin-bottom: 0.5rem;">Awaiting Analysis</h3>
                            <p style="color: var(--text-muted); font-size: 0.95rem;">
                                Enter morphology data to trigger the AI inference engine.
                            </p>
                        </div>
                    `;
                    resultsSection.style.opacity = '1';
                }, 300);
            }
        });
    }
});
