document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultCard = document.getElementById('result-card');
    const hamnosysTags = document.getElementById('hamnosys-tags');
    const hamnosysChars = document.getElementById('hamnosys-chars');
    const playBtn = document.getElementById('play-btn');
    const sigmlStorage = document.getElementById('sigml-storage');

    // Handle Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }

        // UI Updates
        dropZone.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update UI with results
            hamnosysTags.textContent = data.hamnosys_tags;
            hamnosysChars.textContent = data.hamnosys_unicode;
            sigmlStorage.value = data.sigml;
            
            // Hide loading, show results
            loadingIndicator.classList.add('hidden');
            dropZone.classList.remove('hidden');
            resultCard.classList.remove('hidden');

            // Optionally auto-play
            playAnimation();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during processing: ' + error.message);
            loadingIndicator.classList.add('hidden');
            dropZone.classList.remove('hidden');
        });
    }

    playBtn.addEventListener('click', playAnimation);

    // Force WebGL canvas recalculation after load
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 1000);

    function playAnimation() {
        // Trigger resize event to ensure WebGL camera aspect ratio is correctly calculated
        window.dispatchEvent(new Event('resize'));
        
        const sigmlText = sigmlStorage.value;
        if (sigmlText) {
            const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
            const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
            if (sigmlArea && playBtn) {
                sigmlArea.value = sigmlText;
                playBtn.click();
            } else if (typeof CWASA !== 'undefined') {
                CWASA.playSiGMLText(sigmlText, 0);
            }
        } else {
            console.warn("No SiGML text available to play.");
        }
    }
});
