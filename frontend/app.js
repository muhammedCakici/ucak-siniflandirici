// Lokal geliştirmede localhost, deploy'da aynı sunucu kullanılır
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : window.location.origin;

// DOM Elements
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImg     = document.getElementById('previewImg');
const previewFilename = document.getElementById('previewFilename');
const previewFilesize = document.getElementById('previewFilesize');
const clearBtn       = document.getElementById('clearBtn');
const analyzeBtn     = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultSection  = document.getElementById('resultSection');
const errorSection   = document.getElementById('errorSection');
const errorMessage   = document.getElementById('errorMessage');

let selectedFile = null;

// ─── Utility ─────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function show(el) { el.classList.add('visible'); }
function hide(el) { el.classList.remove('visible'); }

function hideAllResults() {
    hide(resultSection);
    hide(errorSection);
    hide(loadingSection);
}

// ─── File Handling ────────────────────────────────────────────────────────────

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Geçersiz dosya', 'Lütfen bir görsel dosyası seçin (JPG, PNG, WebP).');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    show(previewSection);
    hideAllResults();
    dropZone.parentElement.style.display = 'none';
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dropZone.parentElement.style.display = 'block';
    hide(previewSection);
    hideAllResults();
}

// ─── Drag & Drop ──────────────────────────────────────────────────────────────

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    if (!dropZone.contains(e.relatedTarget)) {
        dropZone.classList.remove('drag-over');
    }
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

clearBtn.addEventListener('click', clearSelection);

// ─── Prediction ───────────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    hideAllResults();
    show(loadingSection);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Bilinmeyen bir hata oluştu.');
        }

        hide(loadingSection);
        renderResult(data);
        show(resultSection);
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        hide(loadingSection);
        const msg = err.message.includes('fetch')
            ? 'Sunucuya bağlanılamadı. Backend\'in çalıştığından emin olun: <code>uvicorn main:app --reload</code>'
            : err.message;
        showError('Tahmin başarısız oldu', msg);
    } finally {
        analyzeBtn.disabled = false;
    }
});

// ─── Render Result ────────────────────────────────────────────────────────────

function renderResult(data) {
    const { display_name, predicted_class, confidence, all_predictions } = data;

    // Top prediction
    document.getElementById('topName').textContent    = display_name;
    document.getElementById('topClass').textContent   = predicted_class;

    const confValue = document.getElementById('confValue');
    const confQual  = document.getElementById('confQual');
    const confNum   = document.getElementById('confNum');
    const valueRing = document.getElementById('valueRing');

    // Animate confidence number
    animateNumber(confNum, 0, confidence, 1200, '%');

    // Animate ring
    const circumference = 201;
    const offset = circumference - (confidence / 100) * circumference;
    setTimeout(() => {
        valueRing.style.strokeDashoffset = offset;
    }, 100);

    // Qualifier
    if (confidence >= 80) {
        confQual.textContent = 'Yüksek Güven';
        confQual.className = 'confidence-qualifier high';
    } else if (confidence >= 50) {
        confQual.textContent = 'Orta Güven';
        confQual.className = 'confidence-qualifier medium';
    } else {
        confQual.textContent = 'Düşük Güven';
        confQual.className = 'confidence-qualifier low';
    }

    // Prediction list
    const list = document.getElementById('predictionList');
    list.innerHTML = '';

    all_predictions.forEach((pred, i) => {
        const item = document.createElement('div');
        item.className = 'prediction-item' + (i === 0 ? ' top-item' : '');

        item.innerHTML = `
            <div class="prediction-item-name" title="${pred.display_name}">${pred.class_name.replace(/_/g, ' ')}</div>
            <div class="prediction-bar-wrap">
                <div class="prediction-bar ${i === 0 ? 'top-bar' : 'other-bar'}" style="width:0%"></div>
            </div>
            <div class="prediction-percentage">${pred.probability.toFixed(1)}%</div>
        `;

        list.appendChild(item);

        // Animate bars after mount
        setTimeout(() => {
            item.querySelector('.prediction-bar').style.width = pred.probability + '%';
        }, 100 + i * 60);
    });
}

function animateNumber(el, from, to, duration, suffix = '') {
    const start = performance.now();
    function update(time) {
        const elapsed = time - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3); // cubic ease out
        el.textContent = (from + (to - from) * ease).toFixed(1) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ─── Error Display ────────────────────────────────────────────────────────────

function showError(title, message) {
    document.getElementById('errorTitle').textContent = title;
    document.getElementById('errorMessage').innerHTML = message;
    show(errorSection);
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
