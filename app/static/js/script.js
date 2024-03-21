document.addEventListener('DOMContentLoaded', function() {
    // To check if there's a predicted digit
    const predictedDigit = document.querySelector('#predicted-digit');

    // If there's a predicted digit, reload the page after 5 seconds
    if (predictedDigit.innerText !== 'Predicted Digit: ') {
        setTimeout(() => {
            window.location.reload();
        }, 5000); // 5000 milliseconds = 5 seconds
    }
});
