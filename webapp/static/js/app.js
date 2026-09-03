/**
 * Sign Language to 3D Avatar Converter
 * Synchronized Video & CWASA WebGL Avatar Playback Engine
 * Includes Generated HamNoSys Phonetic Sequence & Keyboard Symbol Breakdown
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM References
    const dropZone            = document.getElementById('drop-zone');
    const fileInput           = document.getElementById('video-input');
    const videoPreviewWrapper = document.getElementById('video-preview-wrapper');
    const inputVideo          = document.getElementById('input-video');
    const videoFilenameTag    = document.getElementById('video-filename-tag');
    const btnChangeVideo      = document.getElementById('btn-change-video');
    const loadingOverlay      = document.getElementById('loading-overlay');

    const fpsBadge         = document.getElementById('fps-badge');
    const frameBadge       = document.getElementById('frame-badge');
    const frameCounterPill = document.getElementById('frame-counter-pill');

    const btnPlaySync   = document.getElementById('btn-play-sync');
    const btnStop       = document.getElementById('btn-stop');
    const btnReplay     = document.getElementById('btn-replay');
    const loopToggle    = document.getElementById('loop-toggle');
    const speedButtons  = document.querySelectorAll('.speed-btn');
    const btnPrevFrame  = document.getElementById('btn-prev-frame');
    const btnNextFrame  = document.getElementById('btn-next-frame');
    const benchmarkBtns = document.querySelectorAll('.benchmark-btn');

    // Generated HamNoSys Sequence Card Elements
    const resultsDrawer         = document.getElementById('results-drawer');
    const hamnosysTagsBox       = document.getElementById('hamnosys-tags-box');
    const hamnosysGlyphsBox     = document.getElementById('hamnosys-glyphs-box');
    const hamnosysBreakdownGrid = document.getElementById('hamnosys-breakdown-grid');
    const btnCopyTokens         = document.getElementById('btn-copy-tokens');
    const copyBtnText           = document.getElementById('copy-btn-text');
    const sigmlStorage          = document.getElementById('sigml-storage');

    // SiGML XML Code Inspector Elements
    const btnCopySigml     = document.getElementById('btn-copy-sigml');
    const copySigmlText    = document.getElementById('copy-sigml-text');
    const sigmlCodeDisplay = document.getElementById('sigml-code-display');

    let currentSigml = '';
    let currentSpeed = 1.0;

    // --- Complete HamNoSys Metadata Dictionary ---
    const HAMNOSYS_META = {
        // Symmetry & Structure
        'hamsymmlr': { category: 'SYMMETRY & STRUCTURE', desc: 'Symmetrical Both Hands' },
        'hamsymmspatial': { category: 'SYMMETRY & STRUCTURE', desc: 'Spatial Symmetry' },
        'hamparal': { category: 'SYMMETRY & STRUCTURE', desc: 'Parallel Hands' },
        'hamreplace': { category: 'STATE TRANSITION', desc: 'Orientation Replace' },

        // Handshape
        'hamflathand': { category: 'HANDSHAPE', desc: 'Flat Handshape' },
        'hamfist': { category: 'HANDSHAPE', desc: 'Fist Handshape' },
        'hamfinger2': { category: 'HANDSHAPE', desc: 'Index Finger Extended' },
        'hamfinger23': { category: 'HANDSHAPE', desc: 'Index & Middle (V-Shape)' },
        'hamfinger23spread': { category: 'HANDSHAPE', desc: 'Index & Middle Spread' },
        'hamfinger2345': { category: 'HANDSHAPE', desc: 'Open 4/5 Fingers Extended' },
        'hamceeall': { category: 'HANDSHAPE', desc: 'C-Handshape (All Fingers)' },
        'hamcee12': { category: 'HANDSHAPE', desc: 'C-Handshape (Index & Thumb)' },
        'hampinchall': { category: 'HANDSHAPE', desc: 'Pinch (All Fingers)' },
        'hampinch12': { category: 'HANDSHAPE', desc: 'Pinch (Index & Thumb)' },
        'hamthumboutmod': { category: 'HANDSHAPE', desc: 'Thumb Outward Modifier' },
        'hamthumbacrossmod': { category: 'HANDSHAPE', desc: 'Thumb Across Palm' },
        'hamthumbopenmod': { category: 'HANDSHAPE', desc: 'Thumb Open' },
        'hamfingerstraightmod': { category: 'HANDSHAPE', desc: 'Straight Finger' },
        'hamfingerbendmod': { category: 'HANDSHAPE', desc: 'Bent Finger' },
        'hamfingerhookmod': { category: 'HANDSHAPE', desc: 'Hooked Finger' },
        'hamdoublebent': { category: 'HANDSHAPE', desc: 'Double Bent Fingers' },
        'hamdoublehooked': { category: 'HANDSHAPE', desc: 'Double Hooked Fingers' },

        // Extended Finger Direction
        'hamextfingeru': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Upward' },
        'hamextfingerd': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Downward' },
        'hamextfingerl': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Leftward' },
        'hamextfingerr': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Rightward' },
        'hamextfingerul': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Up-Left' },
        'hamextfingerur': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Up-Right' },
        'hamextfingerdl': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Down-Left' },
        'hamextfingerdr': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Down-Right' },
        'hamextfingero': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Outward' },
        'hamextfingeri': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Inward' },
        'hamextfingerol': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Out-Left' },
        'hamextfingeror': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Out-Right' },
        'hamextfingeril': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger In-Left' },
        'hamextfingerir': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger In-Right' },
        'hamextfingerui': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Up-In' },
        'hamextfingeruo': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Up-Out' },
        'hamextfingerdi': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Down-In' },
        'hamextfingerdo': { category: 'EXTENDED FINGER DIRECTION', desc: 'Finger Down-Out' },

        // Palm Orientation
        'hampalmu': { category: 'PALM ORIENTATION', desc: 'Palm Up / Towards' },
        'hampalmd': { category: 'PALM ORIENTATION', desc: 'Palm Down' },
        'hampalml': { category: 'PALM ORIENTATION', desc: 'Palm Left' },
        'hampalmr': { category: 'PALM ORIENTATION', desc: 'Palm Right' },
        'hampalmul': { category: 'PALM ORIENTATION', desc: 'Palm Up-Left' },
        'hampalmur': { category: 'PALM ORIENTATION', desc: 'Palm Up-Right' },
        'hampalmdl': { category: 'PALM ORIENTATION', desc: 'Palm Down-Left' },
        'hampalmdr': { category: 'PALM ORIENTATION', desc: 'Palm Down-Right' },

        // Body & Spatial Location
        'hamhead': { category: 'BODY & SPATIAL LOCATION', desc: 'Head Location' },
        'hamheadtop': { category: 'BODY & SPATIAL LOCATION', desc: 'Top of Head' },
        'hamforehead': { category: 'BODY & SPATIAL LOCATION', desc: 'Forehead Location' },
        'hameyebrows': { category: 'BODY & SPATIAL LOCATION', desc: 'Eyebrows Location' },
        'hameyes': { category: 'BODY & SPATIAL LOCATION', desc: 'Eyes Location' },
        'hamnose': { category: 'BODY & SPATIAL LOCATION', desc: 'Nose Location' },
        'hamlips': { category: 'BODY & SPATIAL LOCATION', desc: 'Lips Location' },
        'hammouth': { category: 'BODY & SPATIAL LOCATION', desc: 'Mouth Location' },
        'hamchin': { category: 'BODY & SPATIAL LOCATION', desc: 'Chin Location' },
        'hamunderchin': { category: 'BODY & SPATIAL LOCATION', desc: 'Under Chin' },
        'hamneck': { category: 'BODY & SPATIAL LOCATION', desc: 'Neck Location' },
        'hamshoulders': { category: 'BODY & SPATIAL LOCATION', desc: 'Shoulders Location' },
        'hamchest': { category: 'BODY & SPATIAL LOCATION', desc: 'Chest Location' },
        'hamstomach': { category: 'BODY & SPATIAL LOCATION', desc: 'Stomach Location' },
        'hambelowstomach': { category: 'BODY & SPATIAL LOCATION', desc: 'Below Stomach Location' },
        'hamwristback': { category: 'BODY & SPATIAL LOCATION', desc: 'Back of Wrist' },
        'hamwristpulse': { category: 'BODY & SPATIAL LOCATION', desc: 'Wrist Pulse' },
        'hampalm': { category: 'BODY & SPATIAL LOCATION', desc: 'Palm Location' },
        'hamfingertip': { category: 'BODY & SPATIAL LOCATION', desc: 'Fingertip Location' },
        'hamthumb': { category: 'BODY & SPATIAL LOCATION', desc: 'Thumb Location' },
        'hamindexfinger': { category: 'BODY & SPATIAL LOCATION', desc: 'Index Finger Location' },
        'hammiddlefinger': { category: 'BODY & SPATIAL LOCATION', desc: 'Middle Finger Location' },
        'hamringfinger': { category: 'BODY & SPATIAL LOCATION', desc: 'Ring Finger Location' },
        'hampinky': { category: 'BODY & SPATIAL LOCATION', desc: 'Pinky Finger Location' },
        'hamupperarm': { category: 'BODY & SPATIAL LOCATION', desc: 'Upper Arm Location' },
        'hamelbow': { category: 'BODY & SPATIAL LOCATION', desc: 'Elbow Location' },

        // Contact & Touch
        'hamtouch': { category: 'CONTACT & TOUCH', desc: 'Touch Contact' },
        'hamclose': { category: 'CONTACT & TOUCH', desc: 'Close Proximity' },
        'hambrushing': { category: 'CONTACT & TOUCH', desc: 'Brushing Contact' },
        'haminterlock': { category: 'CONTACT & TOUCH', desc: 'Interlocked Contact' },
        'hamgrasp': { category: 'CONTACT & TOUCH', desc: 'Grasping Contact' },

        // Movement & Motion
        'hammoveu': { category: 'MOVEMENT & MOTION', desc: 'Upward Movement' },
        'hammoved': { category: 'MOVEMENT & MOTION', desc: 'Downward Movement' },
        'hammovel': { category: 'MOVEMENT & MOTION', desc: 'Leftward Movement' },
        'hammover': { category: 'MOVEMENT & MOTION', desc: 'Rightward Movement' },
        'hammoveo': { category: 'MOVEMENT & MOTION', desc: 'Outward Movement' },
        'hammovei': { category: 'MOVEMENT & MOTION', desc: 'Inward Movement' },
        'hammoveul': { category: 'MOVEMENT & MOTION', desc: 'Up-Left Movement' },
        'hammoveur': { category: 'MOVEMENT & MOTION', desc: 'Up-Right Movement' },
        'hammovedl': { category: 'MOVEMENT & MOTION', desc: 'Down-Left Movement' },
        'hammovedr': { category: 'MOVEMENT & MOTION', desc: 'Down-Right Movement' },
        'hammoveol': { category: 'MOVEMENT & MOTION', desc: 'Out-Left Movement' },
        'hammoveor': { category: 'MOVEMENT & MOTION', desc: 'Out-Right Movement' },
        'hammoveil': { category: 'MOVEMENT & MOTION', desc: 'In-Left Movement' },
        'hammoveir': { category: 'MOVEMENT & MOTION', desc: 'In-Right Movement' },
        'hamcircleo': { category: 'MOVEMENT & MOTION', desc: 'Outward Circle' },
        'hamcirclei': { category: 'MOVEMENT & MOTION', desc: 'Inward Circle' },
        'hamcircleu': { category: 'MOVEMENT & MOTION', desc: 'Upward Circle' },
        'hamcircled': { category: 'MOVEMENT & MOTION', desc: 'Downward Circle' },
        'hamcirclel': { category: 'MOVEMENT & MOTION', desc: 'Leftward Circle' },
        'hamcircler': { category: 'MOVEMENT & MOTION', desc: 'Rightward Circle' },
        'hamnodding': { category: 'MOVEMENT & MOTION', desc: 'Nodding Movement' },
        'hamswinging': { category: 'MOVEMENT & MOTION', desc: 'Swinging Movement' },
        'hamtwisting': { category: 'MOVEMENT & MOTION', desc: 'Twisting Movement' },
        'hamwavy': { category: 'MOVEMENT & MOTION', desc: 'Wavy Movement' },
        'hamzigzag': { category: 'MOVEMENT & MOTION', desc: 'Zigzag Movement' },
        'hamfingerplay': { category: 'MOVEMENT & MOTION', desc: 'Finger Play Movement' },
        'hamfast': { category: 'MOVEMENT & MOTION', desc: 'Fast Movement' },
        'hamslow': { category: 'MOVEMENT & MOTION', desc: 'Slow Movement' },
        'hamlargemod': { category: 'MOVEMENT & MOTION', desc: 'Large Motion' },
        'hamsmallmod': { category: 'MOVEMENT & MOTION', desc: 'Small Motion' },
        'hamrepeatreverse': { category: 'MOVEMENT & MOTION', desc: 'Repeat Reverse' },
        'hamrepeatfromstart': { category: 'MOVEMENT & MOTION', desc: 'Repeat From Start' }
    };

    // --- Render HamNoSys Breakdown ---
    function renderHamNoSysBreakdown(tagsString, unicodeString) {
        if (!tagsString) return;

        if (resultsDrawer) {
            resultsDrawer.classList.remove('hidden');
        }
        const sigmlCard = document.getElementById('sigml-inspector-card');
        if (sigmlCard) {
            sigmlCard.classList.remove('hidden');
        }

        if (hamnosysTagsBox) {
            hamnosysTagsBox.textContent = tagsString;
        }
        if (hamnosysGlyphsBox) {
            hamnosysGlyphsBox.textContent = unicodeString || '';
        }
        if (!hamnosysBreakdownGrid) return;

        const tokens = tagsString.trim().split(/\s+/);
        hamnosysBreakdownGrid.innerHTML = '';


        const categoryOrder = [
            'SYMMETRY & STRUCTURE',
            'HANDSHAPE',
            'EXTENDED FINGER DIRECTION',
            'PALM ORIENTATION',
            'BODY & SPATIAL LOCATION',
            'CONTACT & TOUCH',
            'MOVEMENT & MOTION',
            'STATE TRANSITION'
        ];

        const grouped = {};
        tokens.forEach(tok => {
            const meta = HAMNOSYS_META[tok] || {
                category: 'GESTURAL COMPONENT',
                desc: tok.replace(/^ham/, '').replace(/mod$/, ' modifier')
            };
            if (!grouped[meta.category]) grouped[meta.category] = [];
            grouped[meta.category].push({ token: tok, desc: meta.desc });
        });

        const itemsToRender = [];
        categoryOrder.forEach(cat => {
            if (grouped[cat]) {
                grouped[cat].forEach(item => {
                    itemsToRender.push({ category: cat, token: item.token, desc: item.desc });
                });
            }
        });

        // Any extra categories
        Object.keys(grouped).forEach(cat => {
            if (!categoryOrder.includes(cat)) {
                grouped[cat].forEach(item => {
                    itemsToRender.push({ category: cat, token: item.token, desc: item.desc });
                });
            }
        });

        itemsToRender.forEach((item, idx) => {
            const col = document.createElement('div');
            col.className = 'breakdown-item';
            col.innerHTML = `
                <div class="breakdown-item-header">
                    <span class="dot"></span>
                    <span>${item.category}</span>
                </div>
                <div class="breakdown-pill">
                    <span class="breakdown-token">${item.token}</span>
                    <span class="breakdown-desc">(${item.desc})</span>
                </div>
            `;
            hamnosysBreakdownGrid.appendChild(col);

            // Add dotted horizontal divider after every pair in 2-column grid
            if (idx % 2 === 1 && idx < itemsToRender.length - 1) {
                const divider = document.createElement('div');
                divider.className = 'breakdown-row-divider';
                hamnosysBreakdownGrid.appendChild(divider);
            }
        });
    }

    // --- Copy Tokens Button ---
    if (btnCopyTokens) {
        btnCopyTokens.addEventListener('click', () => {
            const text = hamnosysTagsBox ? hamnosysTagsBox.textContent.trim() : '';
            if (text) {
                navigator.clipboard.writeText(text).then(() => {
                    if (copyBtnText) copyBtnText.textContent = 'Copied!';
                    setTimeout(() => {
                        if (copyBtnText) copyBtnText.textContent = 'Copy Tokens';
                    }, 2000);
                });
            }
        });
    }

    // --- SiGML XML Display Updater ---
    function updateSigmlDisplay(sigml) {
        if (!sigmlCodeDisplay) return;
        const codeEl = sigmlCodeDisplay.querySelector('code');
        const content = sigml ? sigml.trim() : DEFAULT_SIGML.trim();
        if (codeEl) {
            codeEl.textContent = content;
        } else {
            sigmlCodeDisplay.textContent = content;
        }
    }

    // --- Copy SiGML XML Button ---
    if (btnCopySigml) {
        btnCopySigml.addEventListener('click', () => {
            const text = currentSigml || (sigmlStorage ? sigmlStorage.value : '');
            if (text) {
                navigator.clipboard.writeText(text).then(() => {
                    if (copySigmlText) copySigmlText.textContent = 'Copied!';
                    setTimeout(() => {
                        if (copySigmlText) copySigmlText.textContent = 'Copy SiGML XML';
                    }, 2000);
                });
            }
        });
    }

    const DEFAULT_SIGML = `<?xml version="1.0" encoding="UTF-8"?>
<sigml>
\t<hns_sign gloss="sign">
\t\t<hamnosys_nonmanual/>
\t\t<hamnosys_manual>
\t\t\t<hamflathand/>
\t\t\t<hamextfingeru/>
\t\t\t<hampalmr/>
\t\t\t<hamheadtop/>
\t\t\t<hamtouch/>
\t\t\t<hammoveu/>
\t\t</hamnosys_manual>
\t</hns_sign>
</sigml>`;

    currentSigml = DEFAULT_SIGML;
    if (sigmlStorage) sigmlStorage.value = DEFAULT_SIGML;

    // Initialize initial default breakdown & SiGML display
    renderHamNoSysBreakdown(
        'hamflathand hamextfingeru hampalmr hamheadtop hamtouch hammoveu',
        'Ã®â‚¬Â¡ Ã®â‚¬Â  Ã®â‚¬Âº Ã®Ââ‚¬ Ã®Ââ€š Ã®â€šâ‚¬'
    );
    updateSigmlDisplay(DEFAULT_SIGML);

    // --- Core Play Functions ---
    // CWASA initialises after the page's load event. Do not rely on clicking
    // its hidden form control: that can leave the avatar in its neutral pose.
    function avatarEngineIsReady() {
        return typeof window.CWASA !== 'undefined'
            && typeof window.CWASA.playSiGMLText === 'function';
    }

    function isPlayableSigml(sigmlText) {
        if (!sigmlText || !sigmlText.trim()) return false;
        const documentNode = new DOMParser().parseFromString(sigmlText, 'application/xml');
        return !documentNode.querySelector('parsererror')
            && documentNode.documentElement
            && documentNode.documentElement.nodeName.toLowerCase() === 'sigml'
            && documentNode.querySelector('hns_sign, sign');
    }

    function playAvatarAnimation() {
        const sigmlText = sigmlStorage && sigmlStorage.value ? sigmlStorage.value : currentSigml;
        if (!isPlayableSigml(sigmlText)) {
            console.error('Avatar playback skipped: the generated SiGML is empty or invalid.');
            return false;
        }

        // Keep this populated for diagnostics, but use CWASA's public API to
        // start playback directly instead of a synthetic click on a hidden button.
        const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
        if (sigmlArea) sigmlArea.value = sigmlText;

        if (avatarEngineIsReady()) {
            window.CWASA.stopSiGML(0);
            window.CWASA.playSiGMLText(sigmlText, 0);
            return true;
        }

        console.warn('Avatar engine is still loading; playback will be retried.');
        return false;
    }

    function stopAvatarAnimation() {
        if (typeof window.CWASA !== 'undefined' && typeof window.CWASA.stopSiGML === 'function') {
            window.CWASA.stopSiGML(0);
        }
    }

    function playBoth() {
        if (inputVideo && inputVideo.src) {
            inputVideo.currentTime = 0;
            inputVideo.playbackRate = currentSpeed;
            inputVideo.play().catch(e => console.log('Video autoplay note:', e));
        }
        // The external avatar bundle can finish initialising slightly after a
        // processed video returns. Retry rather than silently retaining its
        // default standing posture.
        let attempts = 0;
        const startAvatar = () => {
            if (playAvatarAnimation() || attempts++ >= 20) return;
            setTimeout(startAvatar, 150);
        };
        startAvatar();
    }

    function stopBoth() {
        if (inputVideo) {
            inputVideo.pause();
        }
        stopAvatarAnimation();
    }

    function replayBoth() {
        stopBoth();
        setTimeout(() => {
            playBoth();
        }, 150);
    }

    // --- Video Source Loading ---
    function loadVideoSource(src, filename) {
        if (!inputVideo) return;
        inputVideo.src = src;
        inputVideo.playbackRate = currentSpeed;
        if (videoFilenameTag) videoFilenameTag.textContent = filename || 'Video';
        if (dropZone) dropZone.classList.add('hidden');
        if (videoPreviewWrapper) videoPreviewWrapper.classList.remove('hidden');
        inputVideo.load();
    }

    // --- File Upload Handling ---
    function handleFileUpload(file) {
        if (!file.type.startsWith('video/') && !/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {
            alert('Please upload a valid sign language video file.');
            return;
        }

        const localUrl = URL.createObjectURL(file);
        loadVideoSource(localUrl, file.name);

        benchmarkBtns.forEach(b => b.classList.remove('active'));

        // Hide previous results until new analysis finishes
        if (resultsDrawer) resultsDrawer.classList.add('hidden');
        const sigmlCard = document.getElementById('sigml-inspector-card');
        if (sigmlCard) sigmlCard.classList.add('hidden');

        if (loadingOverlay) loadingOverlay.classList.remove('hidden');


        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', { method: 'POST', body: formData })
            .then(res => {
                if (!res.ok) return res.json().then(d => { throw new Error(d.error || 'Server error'); });
                return res.json();
            })
            .then(data => {
                if (loadingOverlay) loadingOverlay.classList.add('hidden');

                renderHamNoSysBreakdown(data.hamnosys_tags, data.hamnosys_unicode);

                if (data.sigml) {
                    currentSigml = data.sigml;
                    if (sigmlStorage) sigmlStorage.value = data.sigml;
                    updateSigmlDisplay(data.sigml);
                }

                // Auto-play both after processing
                setTimeout(() => {
                    playBoth();
                }, 300);
            })
            .catch(err => {
                console.error(err);
                if (loadingOverlay) loadingOverlay.classList.add('hidden');
                alert('Processing error: ' + err.message);
            });
    }

    // --- Drag & Drop Listeners ---
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
        });
    }
    if (fileInput) {
        fileInput.addEventListener('change', e => {
            if (e.target.files.length) handleFileUpload(e.target.files[0]);
        });
    }
    if (btnChangeVideo) {
        btnChangeVideo.addEventListener('click', () => fileInput.click());
    }

    // --- Benchmark Dataset Clips ---
    benchmarkBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const filename = btn.getAttribute('data-file');
            benchmarkBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            loadVideoSource(`/samples/${filename}`, filename);
            if (loadingOverlay) loadingOverlay.classList.remove('hidden');

            fetch(`/process-sample/${encodeURIComponent(filename)}`, { method: 'POST' })
                .then(r => {
                    if (!r.ok) return r.json().then(d => { throw new Error(d.error || 'Server error'); });
                    return r.json();
                })
                .then(data => {
                    if (loadingOverlay) loadingOverlay.classList.add('hidden');
                    renderHamNoSysBreakdown(data.hamnosys_tags, data.hamnosys_unicode);
                    if (data.sigml) {
                        currentSigml = data.sigml;
                        if (sigmlStorage) sigmlStorage.value = data.sigml;
                    }
                    setTimeout(() => {
                        playBoth();
                    }, 300);
                })
                .catch(err => {
                    console.warn(err);
                    if (loadingOverlay) loadingOverlay.classList.add('hidden');
                    playBoth();
                });
        });
    });

    // --- Control Buttons ---
    if (btnPlaySync) btnPlaySync.addEventListener('click', playBoth);
    if (btnStop)     btnStop.addEventListener('click', stopBoth);
    if (btnReplay)   btnReplay.addEventListener('click', replayBoth);

    // --- Continuous Loop ---
    if (inputVideo) {
        inputVideo.addEventListener('ended', () => {
            if (loopToggle && loopToggle.checked) {
                playBoth();
            }
        });
    }

    // --- Speed Buttons ---
    speedButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            speedButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentSpeed = parseFloat(btn.getAttribute('data-speed'));
            if (inputVideo) inputVideo.playbackRate = currentSpeed;

            const logSpd = Math.log2(currentSpeed);
            const txtLog = document.querySelector('.txtLogSpeed.av0');
            if (txtLog) {
                txtLog.value = (logSpd >= 0 ? '+' : '') + logSpd.toFixed(2);
                txtLog.dispatchEvent(new Event('change'));
            }
        });
    });

    // --- Frame Stepping ---
    if (btnNextFrame) {
        btnNextFrame.addEventListener('click', () => {
            stopBoth();
            if (inputVideo && inputVideo.duration) {
                inputVideo.currentTime = Math.min(inputVideo.duration, inputVideo.currentTime + (1 / 25));
            }
            const nextBtn = document.querySelector('.bttnNextF.av0');
            if (nextBtn) nextBtn.click();
        });
    }
    if (btnPrevFrame) {
        btnPrevFrame.addEventListener('click', () => {
            stopBoth();
            if (inputVideo) {
                inputVideo.currentTime = Math.max(0, inputVideo.currentTime - (1 / 25));
            }
            const prevBtn = document.querySelector('.bttnPrevF.av0');
            if (prevBtn) prevBtn.click();
        });
    }

    // --- Telemetry Badge Sync ---
    setInterval(() => {
        const fpsField = document.querySelector('.txtFPS.av0');
        if (fpsField && fpsField.value && fpsField.value !== '00.00') {
            const fps = parseFloat(fpsField.value).toFixed(2);
            if (fpsBadge) fpsBadge.textContent = `FPS: ${fps}`;
        }

        const sfField = document.querySelector('.txtSF.av0');
        if (sfField && sfField.value && sfField.value !== '0/0') {
            if (frameBadge) frameBadge.textContent = `Frame: ${sfField.value}`;
            if (frameCounterPill) frameCounterPill.textContent = sfField.value;
        } else if (inputVideo && inputVideo.duration) {
            const cur = Math.floor(inputVideo.currentTime * 25);
            const tot = Math.floor(inputVideo.duration * 25);
            if (frameBadge) frameBadge.textContent = `Frame: ${cur}/${tot}`;
            if (frameCounterPill) frameCounterPill.textContent = `${cur}/${tot}`;
        }
    }, 200);

    // Initial resize call after 1 second (once only, without disrupting WebGL ticker)
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 1000);
});
