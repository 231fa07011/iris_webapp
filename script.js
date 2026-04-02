
let modelData = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    // 🌸 1. Load Model Data
    try {
        const response = await fetch('model_data.json?v=2.2');
        modelData = await response.json();
        console.log('✅ Botanical Intelligence loaded.');
    } catch (err) {
        console.error('❌ Failed to load model data:', err);
        showError("Neural engine failed to ignite. Check JSON accessibility.");
    }

    // 🌸 2. Progress Bar Animation (Existing logic)
    initUIComponents();
    
    // 🌸 3. Load Local History
    renderHistory();
});

function initUIComponents() {
    // Input visual feedback
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            const min = parseFloat(e.target.min);
            const max = parseFloat(e.target.max);
            
            if (val < min || val > max) {
                input.style.borderColor = 'rgba(239, 68, 68, 0.5)';
                input.style.boxShadow = '0 0 0 4px rgba(239, 68, 68, 0.1)';
            } else {
                input.style.borderColor = '#10B981';
                input.style.boxShadow = '0 0 0 4px rgba(16, 185, 129, 0.1)';
            }
        });
    });

    // Reset handler
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('reset', () => {
            const resultsSection = document.getElementById('results-display');
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
                renderHistory(); // Keep history visible
            }, 300);
        });
    }
}

// --- CORE ML INFERENCE ENGINE ---
async function handlePredict() {
    if (!modelData) return showError("Intelligence engine is still warming up...");

    // 1. Harvest inputs
    const features = [
        parseFloat(document.getElementById('sepal_length').value),
        parseFloat(document.getElementById('sepal_width').value),
        parseFloat(document.getElementById('petal_length').value),
        parseFloat(document.getElementById('petal_width').value)
    ];

    // Basic Validation
    if (features.some(isNaN)) return showError("Please enter valid numerical measurements.");
    if (features.some(v => v < 0 || v > 15)) return showError("Botanical bounds exceeded (0-15cm).");

    // 2. Execute Forest Inference (Ensemble)
    // Each tree in Forest returns a probability distribution (array of 3 floats)
    let aggregatedProbs = [0, 0, 0];
    
    modelData.trees.forEach(tree => {
        const leafResult = walkTree(tree, features);
        // leafResult is typically an array within an array [[count_0, count_1, count_2]] in scikit-learn
        const counts = Array.isArray(leafResult[0]) ? leafResult[0] : leafResult;
        
        // Normalize counts to probabilities (0-1) for this tree
        const sum = counts.reduce((a, b) => a + b, 0);
        const probs = sum > 0 ? counts.map(x => x / sum) : [0, 0, 0];

        aggregatedProbs[0] += probs[0];
        aggregatedProbs[1] += probs[1];
        aggregatedProbs[2] += probs[2];
    });

    // Average the probabilities across all trees
    const treeCount = modelData.trees.length;
    aggregatedProbs = aggregatedProbs.map(v => (v / treeCount) * 100);

    // 3. Determine Winner
    const maxIdx = aggregatedProbs.indexOf(Math.max(...aggregatedProbs));
    const prediction = modelData.target_names[maxIdx];

    // 4. Update UI
    updateResultsUI(prediction, aggregatedProbs);
    
    // 5. Save to History
    saveToLocalHistory(prediction, Math.max(...aggregatedProbs));
}

function walkTree(tree, features) {
    let nodeIndex = 0;
    while (true) {
        const featureIdx = tree.feature[nodeIndex];
        
        // If featureIdx == -2 (sklearn convention for leaf node) or it's a leaf
        if (featureIdx === -2 || tree.children_left[nodeIndex] === -1) {
            return tree.value[nodeIndex];
        }

        const threshold = tree.threshold[nodeIndex];
        if (features[featureIdx] <= threshold) {
            nodeIndex = tree.children_left[nodeIndex];
        } else {
            nodeIndex = tree.children_right[nodeIndex];
        }
    }
}

// --- UI UPDATES ---
function updateResultsUI(prediction, probs) {
    const resultsDisplay = document.getElementById('results-display');
    
    // Sort probs for display
    const speciesData = modelData.target_names.map((name, i) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        prob: probs[i],
        isWinner: name === prediction
    })).sort((a, b) => b.prob - a.prob);

    const iconMap = {
        'setosa': 'flower',
        'versicolor': 'flower-2',
        'virginica': 'leaf'
    };

    const insightMap = {
        'setosa': 'Iris setosa is distinguished by its small petals and broad sepals. It is the most genetically distinct of the three species.',
        'versicolor': 'Iris versicolor, the "Harlequin Iris", presents moderate morphological traits, often acting as a middle-ground in classification.',
        'virginica': 'Iris virginica is known for its robust size and large petals. It typically requires high moisture environments to thrive.'
    };

    let html = `
        <div class="result-header">
            <div class="result-label">The Classified Species is</div>
            <div class="species-name" style="display: flex; align-items: center; justify-content: center; gap: 0.8rem; color: var(--primary);">
                <i data-lucide="${iconMap[prediction] || 'flower'}"></i>
                ${prediction.charAt(0).toUpperCase() + prediction.slice(1)}
            </div>
        </div>

        <div class="prob-container">
            ${speciesData.map(s => `
                <div class="prob-item ${s.isWinner ? 'winner' : ''}">
                    <div class="prob-top">
                        <span class="species-type">${s.name}</span>
                        <span class="percentage">${s.prob.toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${s.prob}%"></div>
                    </div>
                </div>
            `).join('')}
        </div>

        <div class="feature-info" style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.03); border-radius: 1rem; border: 1px solid var(--border);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; color: var(--primary);">
                <i data-lucide="info" style="width: 14px;"></i>
                <span style="font-weight: 600; font-size: 0.85rem; text-transform: uppercase;">Botanical Insight</span>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.6;">
                ${insightMap[prediction]}
            </p>
        </div>

        <p style="margin-top: 1.5rem; font-size: 0.75rem; color: #475569; text-align: center;">
            Edge Decision Engine: 100 parallel trees analyzed.
        </p>
        
        <div id="history-container"></div>
    `;

    resultsDisplay.innerHTML = html;
    
    // Fade in
    resultsDisplay.style.opacity = '0';
    resultsDisplay.style.transform = 'translateY(20px)';
    
    requestAnimationFrame(() => {
        resultsDisplay.style.transition = 'all 0.8s cubic-bezier(0.16, 1, 0.3, 1)';
        resultsDisplay.style.opacity = '1';
        resultsDisplay.style.transform = 'translateY(0)';
        lucide.createIcons();
        renderHistory();
    });
}

function saveToLocalHistory(prediction, confidence) {
    let history = JSON.parse(localStorage.getItem('iris_history') || '[]');
    const entry = {
        prediction,
        confidence,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    history.unshift(entry);
    localStorage.setItem('iris_history', JSON.stringify(history.slice(0, 10)));
}

function renderHistory() {
    const container = document.getElementById('history-container');
    const history = JSON.parse(localStorage.getItem('iris_history') || '[]');
    
    if (!container) {
        // If not in the results layout yet, we'll wait for the next render
        return;
    }

    let html = `
        <div class="history-section" style="margin-top: 2rem;">
            <h4 style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                <i data-lucide="history" style="width: 14px;"></i>
                Recent Discoveries
            </h4>
            <div class="history-list">
                ${history.length ? history.map(h => `
                    <div class="history-entry" style="display: flex; justify-content: space-between; padding: 0.8rem; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border); border-radius: 0.75rem; margin-bottom: 0.5rem; font-size: 0.85rem;">
                        <div style="display: flex; flex-direction: column;">
                            <span style="font-weight: 600; color: var(--primary);">${h.prediction.charAt(0).toUpperCase() + h.prediction.slice(1)}</span>
                            <span style="font-size: 0.7rem; color: var(--text-muted);">${h.timestamp}</span>
                        </div>
                        <div style="text-align: right; font-weight: 700; opacity: 0.8;">
                            ${h.confidence.toFixed(1)}%
                        </div>
                    </div>
                `).join('') : '<p style="font-size: 0.8rem; color: var(--text-muted); text-align: center; opacity: 0.5;">No history found.</p>'}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    lucide.createIcons();
}

function showError(msg) {
    const errCont = document.getElementById('error-container');
    const errText = document.getElementById('error-text');
    if (errCont && errText) {
        errText.innerText = msg;
        errCont.style.display = 'flex';
        setTimeout(() => { errCont.style.display = 'none'; }, 5000);
    }
}
