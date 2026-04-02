const fs = require('fs');

const raw = fs.readFileSync('model_data.json');
const modelData = JSON.parse(raw);

function walkTree(tree, features) {
    let nodeIndex = 0;
    while (true) {
        const featureIdx = tree.feature[nodeIndex];
        
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

const features = [5.1, 3.5, 1.4, 0.2];
let aggregatedProbs = [0, 0, 0];

modelData.trees.forEach(tree => {
    const leafResult = walkTree(tree, features);
    const counts = Array.isArray(leafResult[0]) ? leafResult[0] : leafResult;
    
    // Normalize counts to probabilities (0-1) for this tree
    const sum = counts.reduce((a, b) => a + b, 0);
    const probs = sum > 0 ? counts.map(x => x / sum) : [0, 0, 0];

    aggregatedProbs[0] += probs[0];
    aggregatedProbs[1] += probs[1];
    aggregatedProbs[2] += probs[2];
});

const treeCount = modelData.trees.length;
aggregatedProbs = aggregatedProbs.map(v => (v / treeCount) * 100);

const maxIdx = aggregatedProbs.indexOf(Math.max(...aggregatedProbs));
const prediction = modelData.target_names[maxIdx];

console.log("Prediction:", prediction);
console.log("Probs:", aggregatedProbs);
