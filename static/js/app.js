// 影像處理應用主邏輯 (Core Application Logic)
// 負責核心狀態管理、API 請求發送以及動態參數面板的調度。

// ==========================================
// 1. 全域狀態與 DOM 引用 (Global State & DOM)
// ==========================================
const paramPanel = document.getElementById('paramPanel');
const paramGroups = document.querySelectorAll('.param-group');
const sidebarModuleList = document.getElementById('sidebarModuleList'); 

let currentMode = 'single'; // 運行模式：'single' (單步) 或 'stack' (管線)
let layerStack = [];        // 儲存管線模式下的算子堆疊
let activeLayerId = null;   // 目前正在編輯參數的圖層 ID

// 儲存從伺服器回傳的管線處理中間數據
let currentPipelineData = []; 
let globalInputUrl = '';
let globalOutputUrl = '';
let globalOriginalHistogram = '';
let globalProcessedHistogram = '';
let globalOriginalSpectrum = '';
let globalProcessedSpectrum = '';

// ==========================================
// 2. 參數管理系統 (Parameter Management)
// ==========================================

/**
 * 將 UI 面板上的參數值同步回 layerStack 中對應的圖層物件。
 */
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
                // 根據輸入類型儲存值 (Checkbox 儲存布林值)
                if (input.type === 'checkbox' || input.type === 'radio') {
                    layer.params[input.name] = input.checked;
                } else {
                    layer.params[input.name] = input.value;
                }
            });
        }
    });
}

/**
 * 將指定圖層的參數載入到 UI 面板的 DOM 元素中。
 * @param {Object} layer - 圖層物件
 */
function loadParamsIntoDOM(layer) {
    paramGroups.forEach(group => {
        const types = group.getAttribute('data-types').split(',');
        if (types.includes(layer.type)) {
            const inputs = group.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (!input.name) return;
                
                // 若圖層已有儲存的參數，則還原；否則使用預設值
                if (layer.params[input.name] !== undefined) {
                    if (input.type === 'checkbox' || input.type === 'radio') {
                        input.checked = layer.params[input.name];
                    } else {
                        input.value = layer.params[input.name];
                    }
                } else {
                    // 備援方案：從 HTML 預設屬性中提取初始值
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
                // 手動觸發事件以更新 UI 數值顯示
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            });
        }
    });
}

/**
 * 根據處理類型切換顯示對應的參數組。
 * @param {string} type - 影像處理算子名稱 (如 'binarize')
 */
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
    // 若無可用參數則隱藏整個面板
    paramPanel.style.display = hasVisibleParams ? 'block' : 'none';
}

/**
 * 單步模式專用：更新目前選中算子的參數面板。
 */
function updateParamsForSingleMode() {
    const checkedRadio = document.querySelector('input[name="process_type"]:checked');
    if (checkedRadio) showParamGroupForType(checkedRadio.value);
}

// ==========================================
// 3. API 請求與後端互動 (API & Network)
// ==========================================
const form = document.getElementById('processForm');
const loading = document.getElementById('loadingIndicator');

/**
 * 主處理函數：打包數據、發送 POST 請求並渲染結果。
 */
async function processImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files || fileInput.files.length === 0) return; 

    const formData = new FormData(form);
    
    // 管線模式下，需將整串 layerStack 序列化後發送
    if (currentMode === 'stack') {
        saveCurrentParamsToLayer();
        formData.append('pipeline_sequence', JSON.stringify(layerStack));
    }
    
    loading.classList.add('show');
    try {
        const response = await fetch('/tool/image-processing', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "伺服器傳回錯誤");

        if (data.mode === 'stack') {
            // [管線模式] 儲存全局與中間節點數據
            currentPipelineData = data.nodes_data;
            globalInputUrl = "data:image/jpeg;base64," + data.global_input;
            globalOutputUrl = "data:image/jpeg;base64," + data.global_output;
            globalOriginalHistogram = data.original_histogram;
            globalProcessedHistogram = data.processed_histogram;
            globalOriginalSpectrum = data.original_spectrum;
            globalProcessedSpectrum = data.processed_spectrum;
            
            // 根據目前選中的圖層決定顯示全局還是節點視圖
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
            // [單步模式] 直接更新主預覽區
            if (data.original_image) document.getElementById('originalImage').src = "data:image/jpeg;base64," + data.original_image;
            document.getElementById('processedImage').src = "data:image/jpeg;base64," + data.processed_image;
            document.getElementById('originalHistogram').src = "data:image/png;base64," + data.original_histogram;
            document.getElementById('processedHistogram').src = "data:image/png;base64," + data.processed_histogram;
            document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + data.original_spectrum;
            document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + data.processed_spectrum;
            
            document.getElementById('histogramSection').style.display = 'block';
            document.getElementById('spectrumSection').style.display = 'block'; 

            // 動態生成並顯示中間處理步驟 (Steps)
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

/**
 * 防抖函數 (Debounce)：避免頻繁拖動滑桿導致 API 請求過載。
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const debouncedProcessImage = debounce(processImage, 500);

// ==========================================
// 4. 事件監聽 (Event Listeners)
// ==========================================

// A. 參數變動監聽：滑桿、選單、勾選框、隱藏欄位
document.querySelectorAll('input[type="range"], select, input[type="checkbox"], input[type="hidden"]').forEach(el => {
    el.addEventListener('input', debouncedProcessImage);
    el.addEventListener('change', debouncedProcessImage);
});

// B. 參數重置功能
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
    
    // 🌟 新增：重置自訂矩陣
    const customKernelInput = document.getElementById('customKernelInput');
    if (customKernelInput) {
        customKernelInput.value = customKernelInput.defaultValue;
        customKernelInput.dispatchEvent(new Event('change'));
    }
});

// C. 圖片檔案上傳監聽
document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
        const origImgTag = document.getElementById('originalImage');

        if (isTiff) {
            // TIFF 特別處理：不支援瀏覽器即時預覽，顯示處理中提示
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

        // 清空舊結果
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

// D. Otsu 自動門檻與滑桿鎖定
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

// E. 模式切換監聽 (單步 vs 管線)
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

// F. 單步模式算子切換
document.querySelectorAll('input[name="process_type"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (currentMode === 'single') {
            updateParamsForSingleMode(); 
            debouncedProcessImage();     
        }
    });
});

/**
 * 側邊欄抽屜開關 (僅適用於特定佈局)
 */
function toggleDrawer() {
    const drawer = document.getElementById('modeDrawer');
    if(drawer) drawer.classList.toggle('open');
}

// ==========================================
// 🌟 G. 自訂矩陣按鈕互動邏輯 (動態適應 Kernel Size)
// ==========================================

/**
 * 動態生成自訂矩陣的 UI 按鈕
 * @param {number} size - 矩陣大小 (如 3, 5, 7)
 * @param {Array} existingMatrix - 既有的矩陣資料 (可選)
 */
function renderCustomKernelGrid(size, existingMatrix = null) {
    const grid = document.getElementById('customKernelGrid');
    const hiddenInput = document.getElementById('customKernelInput');
    if (!grid || !hiddenInput) return;

    // 動態設定 CSS Grid 的欄數
    grid.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
    grid.innerHTML = '';

    let matrix = [];
    for (let i = 0; i < size; i++) {
        let row = [];
        for (let j = 0; j < size; j++) {
            // 若有既有資料則帶入，否則預設為 1
            let val = (existingMatrix && existingMatrix[i] && existingMatrix[i][j] !== undefined) ? existingMatrix[i][j] : 1;
            row.push(val);

            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'kernel-btn';
            btn.innerText = val;
            btn.style.background = val === 1 ? 'rgba(100, 255, 218, 0.2)' : 'rgba(255, 255, 255, 0.05)';
            btn.style.border = '1px solid rgba(100, 255, 218, 0.5)';
            btn.style.color = val === 1 ? '#64ffda' : '#888';
            btn.style.cursor = 'pointer';
            
            // 根據大小動態調整 padding 與字體，避免格子太大擠破版面
            btn.style.padding = size > 7 ? '2px 0' : '8px 0';
            btn.style.fontSize = size > 7 ? '0.7rem' : '1rem';
            btn.style.borderRadius = '4px';
            btn.style.fontWeight = 'bold';
            btn.style.transition = 'all 0.2s';

            grid.appendChild(btn);
        }
        matrix.push(row);
    }
    // 更新隱藏欄位 (這裡不觸發 change，避免無窮迴圈)
    hiddenInput.value = JSON.stringify(matrix);
}

// 1. 網頁載入時，初始化預設 3x3 矩陣
document.addEventListener("DOMContentLoaded", () => {
    renderCustomKernelGrid(3);
});

// 2. 監聽 Kernel Size 滑桿變動，動態重繪矩陣
const ksizeInput = document.querySelector('input[name="ksize"]');
if (ksizeInput) {
    ksizeInput.addEventListener('input', function(e) {
        const newSize = parseInt(e.target.value);
        const hiddenInput = document.getElementById('customKernelInput');
        if (hiddenInput) {
            const currentMatrix = JSON.parse(hiddenInput.value);
            // 只有當大小真的改變時才重繪
            if (currentMatrix.length !== newSize) {
                renderCustomKernelGrid(newSize);
                // 觸發 API 更新
                hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
}

// 3. 監聽隱藏欄位的變化 (例如從管線模式載入舊參數時，還原矩陣畫面)
const customKernelInput = document.getElementById('customKernelInput');
if (customKernelInput) {
    customKernelInput.addEventListener('change', function(e) {
        try {
            const matrix = JSON.parse(e.target.value);
            const currentSize = matrix.length;
            const grid = document.getElementById('customKernelGrid');
            
            // 檢查目前的按鈕數量是否與 matrix 大小相符
            if (grid && grid.children.length !== currentSize * currentSize) {
                renderCustomKernelGrid(currentSize, matrix);
            } else {
                // 數量相符，只更新顏色
                const btns = grid.querySelectorAll('.kernel-btn');
                btns.forEach((btn, idx) => {
                    let r = Math.floor(idx / currentSize);
                    let c = idx % currentSize;
                    let val = matrix[r][c];
                    btn.innerText = val.toString();
                    btn.style.background = val === 1 ? 'rgba(100, 255, 218, 0.2)' : 'rgba(255, 255, 255, 0.05)';
                    btn.style.color = val === 1 ? '#64ffda' : '#888';
                });
            }
        } catch(err) {
            console.error("解析自訂矩陣失敗", err);
        }
    });
}

// 4. 監聽矩陣按鈕點擊事件 (使用事件委派)
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('kernel-btn')) {
        const btn = e.target;
        
        // 切換 0 與 1 的狀態與樣式
        btn.innerText = btn.innerText === '1' ? '0' : '1';
        btn.style.background = btn.innerText === '1' ? 'rgba(100, 255, 218, 0.2)' : 'rgba(255, 255, 255, 0.05)';
        btn.style.color = btn.innerText === '1' ? '#64ffda' : '#888';
        
        // 重新計算並更新隱藏的 input 值
        const wrapper = btn.closest('#customKernelWrapper');
        const btns = wrapper.querySelectorAll('.kernel-btn');
        
        // 動態取得目前的矩陣維度
        const size = Math.sqrt(btns.length);
        let matrix = Array.from({length: size}, () => Array(size).fill(0));
        
        btns.forEach((b, idx) => {
            let r = Math.floor(idx / size);
            let c = idx % size;
            matrix[r][c] = parseInt(b.innerText);
        });
        
        // 更新隱藏欄位，並手動觸發 change 事件讓系統發送 API 請求
        const hiddenInput = document.getElementById('customKernelInput');
        if (hiddenInput) {
            hiddenInput.value = JSON.stringify(matrix);
            hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});
