document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('image-upload');
    const submitButton = document.getElementById('submit-button');
    const refreshButton = document.getElementById('refresh-button');
    const predictedDigit = document.getElementById('predicted-digit');

    submitButton.addEventListener('click', function() {
        // To ensure that a file is selected
        if (!imageUpload.files || !imageUpload.files[0]) {
            alert('Please select an image to submit.');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageUpload.files[0]); // Use 'file' as the key, matching the Flask server

        // Send the image data to the server for prediction
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Prediction request failed.');
            }
            return response.json();
        })
        .then(data => {
            // Update the UI with the predicted digit
            predictedDigit.innerText = 'Predicted Digit: ' + data.digit;
        })
        .catch(error => {
            // Handle errors
            console.error('Prediction error:', error);
            predictedDigit.innerText = 'Prediction failed.';
        });
    });

    refreshButton.addEventListener('click', function() {
        // Reset the predicted digit and clear the file input
        predictedDigit.innerText = 'Predicted Digit: ';
        imageUpload.value = ''; 
    });
});
