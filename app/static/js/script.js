document.addEventListener('DOMContentLoaded', function() {
    // Check if there's a predicted digit
    const predictedDigit = document.querySelector('p');

    // If there's a predicted digit, reload the page after 5 seconds
    if (predictedDigit) {
        setTimeout(() => {
            window.location.reload();
        }, 5000); // 5000 milliseconds = 5 seconds
    }
});
