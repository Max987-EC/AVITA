// 負責核心狀態、API 請求與參數管理

// ==========================================
// 1. 全域變數與 DOM 元素 (Global State & DOM)
// ==========================================
const paramPanel = document.getElementById('paramPanel');
const paramGroups = document.querySelectorAll('.param-group');
const sidebarModuleList = document.getElementById('sidebarModuleList'); 

let currentMode = 'single'; 
let layerStack = [];        
let activeLayerId = null;

let currentPipelineData = []; 
let globalInputUrl = '';
let globalOutputUrl = '';
let globalOriginalHistogram = '';
let globalProcessedHistogram = '';
let globalOriginalSpectrum = '';
let globalProcessedSpectrum = '';

// ==========================================
// 2. 參數儲存與讀取 (Parameter Management)
// ==========================================
function saveCurrentParamsToLayer() {
    if (!activeLayerId) return;
    const layer = layerStack.find(l => l.id === activeLayerId);
    if (!layer) return;

    paramGroups.forEach(group => {
        const types = group.getAttribute('data-types').split(',');
        if (types.includes(layer.type)) {
            const inputs = group.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (!input.name) return; 
                if (input.type === 'checkbox' || input.type === 'radio') {
                    layer.params[input.name] = input.checked;
                } else {
                    layer.params[input.name] = input.value;
                }
            });
        }
    });
}

function loadParamsIntoDOM(layer) {
    paramGroups.forEach(group => {
        const types = group.getAttribute('data-types').split(',');
        if (types.includes(layer.type)) {
            const inputs = group.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (!input.name) return;
                
                if (layer.params[input.name] !== undefined) {
                    if (input.type === 'checkbox' || input.type === 'radio') {
                        input.checked = layer.params[input.name];
                    } else {
                        input.value = layer.params[input.name];
                    }
                } else {
                    if (input.type === 'checkbox' || input.type === 'radio') {
                        input.checked = input.defaultChecked;
                        layer.params[input.name] = input.defaultChecked;
                    } else if (input.tagName.toLowerCase() === 'select') {
                        const defaultOption = Array.from(input.options).find(opt => opt.defaultSelected);
                        const defVal = defaultOption ? defaultOption.value : input.options[0].value;
                        input.value = defVal;
                        layer.params[input.name] = defVal;
                    } else {
                        input.value = input.defaultValue;
                        layer.params[input.name] = input.defaultValue;
                    }
                }
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            });
        }
    });
}

function showParamGroupForType(type) {
    let hasVisibleParams = false;
    paramGroups.forEach(group => {
        const types = group.getAttribute('data-types').split(',');
        if (types.includes(type)) {
            group.style.display = 'block';
            hasVisibleParams = true;
        } else {
            group.style.display = 'none';
        }
    });
    paramPanel.style.display = hasVisibleParams ? 'block' : 'none';
}

function updateParamsForSingleMode() {
    const checkedRadio = document.querySelector('input[name="process_type"]:checked');
    if (checkedRadio) showParamGroupForType(checkedRadio.value);
}

// ==========================================
// 3. API 請求與防抖 (API & Debounce)
// ==========================================
const form = document.getElementById('processForm');
const loading = document.getElementById('loadingIndicator');

async function processImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files || fileInput.files.length === 0) return; 

    const formData = new FormData(form);
    
    if (currentMode === 'stack') {
        saveCurrentParamsToLayer();
        formData.append('pipeline_sequence', JSON.stringify(layerStack));
    }
    
    loading.classList.add('show');
    try {
        const response = await fetch('/tool/image-processing', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "伺服器錯誤");

        if (data.mode === 'stack') {
            currentPipelineData = data.nodes_data;
            globalInputUrl = "data:image/jpeg;base64," + data.global_input;
            globalOutputUrl = "data:image/jpeg;base64," + data.global_output;
            globalOriginalHistogram = data.original_histogram;
            globalProcessedHistogram = data.processed_histogram;
            globalOriginalSpectrum = data.original_spectrum;
            globalProcessedSpectrum = data.processed_spectrum;
            
            if (activeLayerId) {
                const activeIndex = layerStack.findIndex(l => l.id === activeLayerId);
                if (activeIndex !== -1 && currentPipelineData[activeIndex]) {
                    renderNodeView(activeIndex);
                } else {
                    renderGlobalView();
                }
            } else {
                renderGlobalView();
            }
        } else {
            if (data.original_image) document.getElementById('originalImage').src = "data:image/jpeg;base64," + data.original_image;
            document.getElementById('processedImage').src = "data:image/jpeg;base64," + data.processed_image;
            document.getElementById('originalHistogram').src = "data:image/png;base64," + data.original_histogram;
            document.getElementById('processedHistogram').src = "data:image/png;base64," + data.processed_histogram;
            document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + data.original_spectrum;
            document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + data.processed_spectrum;
            
            document.getElementById('histogramSection').style.display = 'block';
            document.getElementById('spectrumSection').style.display = 'block'; 

            const stepsContainer = document.getElementById('stepsContainer');
            const stepsGrid = document.getElementById('stepsGrid');
            stepsGrid.innerHTML = ''; 
            
            if (data.steps && data.steps.length > 0) {
                stepsContainer.style.display = 'block'; 
                data.steps.forEach(step => {
                    const stepBox = document.createElement('div');
                    stepBox.className = 'step-box';
                    stepBox.innerHTML = `
                        <h5>[ ${step.name} ]</h5>
                        <img src="data:image/jpeg;base64,${step.image}" alt="${step.name}">
                    `;
                    stepsGrid.appendChild(stepBox);
                });
            } else {
                stepsContainer.style.display = 'none'; 
            }
        }
    } catch (error) {
        console.error('處理失敗：' + error.message);
    } finally {
        loading.classList.remove('show');
    }
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const debouncedProcessImage = debounce(processImage, 500);

// ==========================================
// 4. 事件監聽綁定 (Event Listeners)
// ==========================================
document.querySelectorAll('input[type="range"], select, input[type="checkbox"]').forEach(el => {
    el.addEventListener('input', debouncedProcessImage);
    el.addEventListener('change', debouncedProcessImage);
});

document.getElementById('resetParamsBtn').addEventListener('click', () => {
    document.querySelectorAll('#paramPanel input[type="range"]').forEach(range => {
        range.value = range.defaultValue;
        range.dispatchEvent(new Event('input'));
    });
    document.querySelectorAll('#paramPanel input[type="checkbox"]').forEach(cb => {
        cb.checked = cb.defaultChecked;
        cb.dispatchEvent(new Event('change'));
    });
    document.querySelectorAll('#paramPanel select').forEach(select => {
        const defaultOption = Array.from(select.options).find(opt => opt.defaultSelected);
        select.value = defaultOption ? defaultOption.value : select.options[0].value;
        select.dispatchEvent(new Event('change'));
    });
});

document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
        const origImgTag = document.getElementById('originalImage');

        if (isTiff) {
            origImgTag.src = "";
            origImgTag.alt = "[ TIFF 格式無法即時預覽，處理中... ]";
            document.getElementById('imageGrid').style.display = 'grid';
        } else {
            const reader = new FileReader();
            reader.onload = function(e) {
                origImgTag.src = e.target.result;
                origImgTag.alt = "原圖預覽";
                document.getElementById('imageGrid').style.display = 'grid';
            }
            reader.readAsDataURL(file);
        }

        document.getElementById('processedImage').src = "";
        document.getElementById('originalHistogram').src = "";
        document.getElementById('processedHistogram').src = "";
        document.getElementById('originalSpectrum').src = ""; 
        document.getElementById('processedSpectrum').src = ""; 
        document.getElementById('histogramSection').style.display = 'none';
        document.getElementById('spectrumSection').style.display = 'none'; 
        document.getElementById('stepsContainer').style.display = 'none';

        processImage();
    }
});

const otsuCheckbox = document.getElementById('otsuCheckbox');
if (otsuCheckbox) {
    otsuCheckbox.addEventListener('change', function(e) {
        const slider = document.getElementById('thresholdSlider');
        if (e.target.checked) {
            slider.disabled = true;
            slider.style.opacity = '0.3';
        } else {
            slider.disabled = false;
            slider.style.opacity = '1';
        }
        debouncedProcessImage();
    });
}

document.querySelectorAll('input[name="app_mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        currentMode = e.target.value;
        const layerManager = document.getElementById('layerManager');

        if (currentMode === 'stack') {
            layerManager.style.display = 'flex';
            if(sidebarModuleList) sidebarModuleList.style.display = 'none';
            
            if(layerStack.length === 0) {
                paramPanel.style.display = 'none';
            } else {
                selectLayer(layerStack[layerStack.length - 1].id);
            }
            renderTopLayers();
            debouncedProcessImage(); 
        } else {
            layerManager.style.display = 'none';
            if(sidebarModuleList) sidebarModuleList.style.display = 'block';
            updateParamsForSingleMode(); 
            debouncedProcessImage();
        }
    });
});

document.querySelectorAll('input[name="process_type"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (currentMode === 'single') {
            updateParamsForSingleMode(); 
            debouncedProcessImage();     
        }
    });
});

function toggleDrawer() {
    const drawer = document.getElementById('modeDrawer');
    if(drawer) drawer.classList.toggle('open');
}
