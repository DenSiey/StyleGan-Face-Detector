document.getElementById('select_img').addEventListener('change', function(event) {
    const input = event.target;
    const imgElement = document.getElementById('img_url');

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            imgElement.src = e.target.result;
        };

        reader.readAsDataURL(input.files[0]);
    }
});

document.addEventListener('DOMContentLoaded', function () {
    const loaderDiv = document.getElementById('loader_div');
    const imgForm = document.getElementById('img_form');
    const detectButton = document.getElementById('detect');
    const label = document.getElementById('browse_img');

    imgForm.addEventListener('submit', function (event) {
        // Show the loader when the form is submitted
        loaderDiv.style.display = 'flex';

        // Disable the Detect button
        detectButton.disabled = true;

        // Prevent the click event on the label
        label.style.pointerEvents = 'none';
    });

    // Hide the loader and enable the Detect button when the page is fully loaded
    window.addEventListener('load', function () {
        loaderDiv.style.display = 'none';
        detectButton.disabled = false;

        // Re-enable the label for future clicks
        label.style.pointerEvents = 'auto';
    });
});

document.addEventListener('DOMContentLoaded', function () {
    // Get the label and the associated input element
    var label = document.getElementById('browse_img');
    var input = document.getElementById('select_img');

    // Add a click event listener to the label
    label.addEventListener('click', function () {
        // Trigger a click on the associated input element
        input.click();
    });
});