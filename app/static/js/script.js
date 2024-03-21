document.addEventListener('DOMContentLoaded', function() {
    // To check if there's a predicted digit
    const predictedDigit = document.querySelector('#predicted-digit');

    // Check if predicted digit element exists and its innerText is not empty
    if (predictedDigit && predictedDigit.innerText.trim() !== 'Predicted Digit:') {
        // Reload the page after 5 seconds
        setTimeout(() => {
            window.location.reload();
        }, 5000); // 5000 milliseconds = 5 seconds
    }
});
