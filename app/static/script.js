document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('image-upload');
    const submitButton = document.getElementById('submit-button');
    const refreshButton = document.getElementById('refresh-button');
    const predictedDigit = document.getElementById('predicted-digit');

    submitButton.addEventListener('click', function() {
        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictedDigit.innerText = 'Predicted Digit: ' + data.prediction;
        })
        .catch(error => {
            console.error('Error:', error);
            predictedDigit.innerText = 'Prediction failed. Please try again.';
        });
    });

    refreshButton.addEventListener('click', function() {
        predictedDigit.innerText = 'Predicted Digit: ';
        imageUpload.value = ''; // Clear the file input
    });
});
