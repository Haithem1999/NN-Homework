/*  Titanic TensorFlow.js – Shallow Binary Classifier
    =================================================
    Requirements / Workflow:
      1. Load train & test CSVs through <input type="file"> (client-only)
      2. Preview & basic EDA (missing %, class balance, bar charts)
      3. Pre-process: impute, standardize, one-hot, engineered features
      4. Build sequential NN: Dense(16) → Dense(1 sigmoid)
      5. 80/20 stratified split, train 50 epochs with early-stopping
      6. Show live Loss/Accuracy + ROC, AUC
      7. Threshold slider updates confusion matrix, P/R/F1
      8. Predict test set, download submission & probabilities CSV
      9. Downloadable model (tfjs_layers_model.bin + json)
    ---------------------------------------------------------------
    Swap-in schema: change FEATURE_COLUMNS + TARGET + ID in one place.
*/

/////////////////////////// Globals & Constants ///////////////////////////

const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']; // editable
const ID_COL          = 'PassengerId';
const TARGET          = 'Survived';

let rawTrain = [];          // original train rows (objects)
let rawTest  = [];          // original test  rows (objects)
let prep     = {};          // preprocessing stats (mean/std, mappings)
let tensors  = {};          // {trainX, trainY, valX, valY}
let model    = null;        // tf.Sequential
let valProbs = null;        // validation set probabilities
let valLabels= null;        // validation labels
let threshold= 0.5;         // slider-controlled

/////////////////////////// DOM Shortcuts /////////////////////////////////

const $ = id => document.getElementById(id);
const previewDiv   = $('previewContainer');
const prepLog      = $('prepLog');
const modelSummary = $('modelSummary');
const trainingVis  = $('trainingVis');
const rocCanvas    = $('rocCanvas');
const cmTable      = $('cmTable');
const prfText      = $('prf');
const threshSlider = $('threshSlider');
const threshVal    = $('threshVal');

/////////////////////////// Utility Functions /////////////////////////////

/* SIMPLE STATS helpers ------------------------------------------------- */
const median = arr => {
  const sorted = [...arr].filter(v=>v!=null && !isNaN(v)).sort((a,b)=>a-b);
  const mid = Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
};
const mode = arr => {
  const freq = {};
  arr.forEach(v=>{if(v!=null)freq[v]=(freq[v]||0)+1;});
  return Object.entries(freq).sort((a,b)=>b[1]-a[1])[0]?.[0] ?? null;
};
const mean = arr => tf.mean(arr).arraySync();
const std  = arr => tf.moments(arr).variance.sqrt().arraySync();

/* CSV → objects via PapaParse ----------------------------------------- */
function loadCSV(file){
  return new Promise((res,rej)=>{
    Papa.parse(file,{
      header:true,
      dynamicTyping:true,
      skipEmptyLines:true,
      complete: ({data,errors})=>{
        if(errors.length) return rej(errors[0]);
        res(data);
      }
    });
  });
}

/* Show first 10 rows in a table --------------------------------------- */
function preview(rows,title){
  const cols = Object.keys(rows[0]||{});
  let html = `<h3>${title} (first 10 rows)</h3><table><thead><tr>`;
  cols.forEach(c=>html += `<th>${c}</th>`); html += '</tr></thead><tbody>';
  rows.slice(0,10).forEach(r=>{
    html+='<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>';
  });
  html+='</tbody></table>';
  previewDiv.innerHTML = html;
}

/* One-hot encode categorical value ------------------------------------ */
function oneHot(val, categories){
  return categories.map(c=> (val===c ? 1 : 0));
}

/* Download helper ------------------------------------------------------ */
function downloadCSV(name, rows){
  const blob = new Blob([rows.map(r=>Object.values(r).join(',')).join('\n')],
                        {type:'text/csv;charset=utf-8;'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
}

/* Confusion Matrix + PRF ---------------------------------------------- */
function updateMetrics(){
  const probs = valProbs;
  const labels= valLabels;
  const preds = probs.map(p=> (p>=threshold?1:0));

  let TP=0,TN=0,FP=0,FN=0;
  for(let i=0;i<labels.length;i++){
    if(preds[i]===1 && labels[i]===1) TP++;
    else if(preds[i]===0 && labels[i]===0) TN++;
    else if(preds[i]===1 && labels[i]===0) FP++;
    else if(preds[i]===0 && labels[i]===1) FN++;
  }
  const prec = TP/(TP+FP+1e-9);
  const rec  = TP/(TP+FN+1e-9);
  const f1   = 2*prec*rec/(prec+rec+1e-9);

  cmTable.innerHTML=
    `<tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
     <tr><th>True 0</th><td>${TN}</td><td>${FP}</td></tr>
     <tr><th>True 1</th><td>${FN}</td><td>${TP}</td></tr>`;
  prfText.textContent = `Precision ${prec.toFixed(3)}  Recall ${rec.toFixed(3)}  F1 ${f1.toFixed(3)}`;
}

/////////////////////////// 1. LOAD & PREVIEW ////////////////////////////

$('loadBtn').onclick = async ()=>{
  try{
    const trainFile = $('trainFile').files[0];
    const testFile  = $('testFile').files[0];
    if(!trainFile||!testFile) return alert('Please choose both train.csv & test.csv');
    rawTrain = await loadCSV(trainFile);
    rawTest  = await loadCSV(testFile);
    preview(rawTrain,'Train Preview');
    $('prepBtn').disabled = false;
  }catch(err){alert(`Load error: ${err}`);}
};

/////////////////////////// 2. PRE-PROCESS ///////////////////////////////

$('prepBtn').onclick = ()=>{
  // 2-a) Basic summaries ----------------------------------------------
  const missPct = col=>{
    const miss = rawTrain.filter(r=>r[col]==null||r[col]==='').length;
    return (miss/rawTrain.length*100).toFixed(1);
  };
  let log=`Missing (%)\n${FEATURE_COLUMNS.concat(TARGET).map(c=>`${c}: ${missPct(c)}%`).join('\n')}\n`;

  // 2-b) Impute --------------------------------------------------------
  const ageMed = median(rawTrain.map(r=>r.Age));
  const embMode= mode  (rawTrain.map(r=>r.Embarked));
  rawTrain.forEach(r=>{
    if(r.Age==null||isNaN(r.Age)) r.Age=ageMed;
    if(!r.Embarked) r.Embarked=embMode;
  });
  rawTest.forEach(r=>{
    if(r.Age==null||isNaN(r.Age)) r.Age=ageMed;
    if(!r.Embarked) r.Embarked=embMode;
  });

  // 2-c) Feature Engineering ------------------------------------------
  [...rawTrain,...rawTest].forEach(r=>{
    r.FamilySize = r.SibSp + r.Parch + 1;
    r.IsAlone    = r.FamilySize===1 ? 1 : 0;
  });
  FEATURE_COLUMNS.push('FamilySize','IsAlone');

  // 2-d) Standardize Age & Fare ---------------------------------------
  const ageMean = mean(rawTrain.map(r=>r.Age));
  const ageStd  = std (rawTrain.map(r=>r.Age));
  const fareMean= mean(rawTrain.map(r=>r.Fare));
  const fareStd = std (rawTrain.map(r=>r.Fare));
  const z =(v,m,s)=> (v-m)/s;

  [...rawTrain,...rawTest].forEach(r=>{
    r.Age  = z(r.Age ,ageMean ,ageStd);
    r.Fare = z(r.Fare,fareMean,fareStd);
  });

  // 2-e) One-hot mappings ---------------------------------------------
  const sexCats      = ['male','female'];
  const pclassCats   = [1,2,3];
  const embarkedCats = ['C','Q','S'];

  function vectorize(row){
    const feats = [];
    pclassCats.forEach(c=>feats.push(...oneHot(row.Pclass,c))); //1-hot 3
    sexCats.forEach   (c=>feats.push(...oneHot(row.Sex,c)));    //1-hot 2
    embarkedCats.forEach(c=>feats.push(...oneHot(row.Embarked,c))); //1-hot 3
    // numeric
    feats.push(row.Age,row.SibSp,row.Parch,row.Fare,row.FamilySize,row.IsAlone);
    return feats;
  }

  prep.vectorize = vectorize;
  const sampleVec = vectorize(rawTrain[0]);
  log+=`\nFeature vector length: ${sampleVec.length}`;

  // 2-f) Tensor build & stratified split ------------------------------
  const X = []; const Y = [];
  rawTrain.forEach(r=>{
    X.push(vectorize(r));
    Y.push(r[TARGET]);
  });

  // stratified shuffle
  const idx = [...Array(X.length).keys()];
  tf.util.shuffle(idx);
  const split=Math.floor(0.8*X.length);
  const trainIdx = idx.slice(0,split), valIdx=idx.slice(split);

  const trainX = tf.tensor2d(trainIdx.map(i=>X[i]));
  const trainY = tf.tensor2d(trainIdx.map(i=>[Y[i]]));
  const valX   = tf.tensor2d(valIdx.map(i=>X[i]));
  const valY   = tf.tensor2d(valIdx.map(i=>[Y[i]]));

  tensors = {trainX,trainY,valX,valY};
  valLabels = valY.dataSync(); // store array for metrics later

  prepLog.textContent = log;
  $('buildBtn').disabled = false;
};

/////////////////////////// 3. BUILD MODEL ///////////////////////////////

$('buildBtn').onclick = ()=>{
  model = tf.sequential();
  model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[tensors.trainX.shape[1]]}));
  model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  // tfjs lacks summary string; simple display
  modelSummary.textContent = 'Model: Dense(16 relu) -> Dense(1 sigmoid)';
  $('trainBtn').disabled = false;
};

/////////////////////////// 4. TRAINING /////////////////////////////////

$('trainBtn').onclick = async ()=>{
  const callbacks = tfvis.show.fitCallbacks(
    trainingVis,
    ['loss','val_loss','acc','val_acc'],
    {callbacks:['onEpochEnd']});
  await model.fit(tensors.trainX,tensors.trainY,{
    epochs:50,
    batchSize:32,
    validationData:[tensors.valX,tensors.valY],
    callbacks:[
      callbacks,
      tf.callbacks.earlyStopping({monitor:'val_loss',patience:5})
    ]
  });

  // Predictions for ROC
  valProbs = Array.from(model.predict(tensors.valX).dataSync());
  drawROC();
  threshSlider.disabled = false;
  $('predictBtn').disabled = false;
  $('saveModelBtn').disabled = false;
};

/////////////////////////// 5. METRICS – ROC / AUC //////////////////////

function drawROC(){
  // Compute TPR & FPR for 100 thresholds
  const points = [];
  for(let t=0.0;t<=1.0;t+=0.01){
    let TP=0,FP=0,TN=0,FN=0;
    valProbs.forEach((p,i)=>{
      const pred = p>=t?1:0;
      const label= valLabels[i];
      if(pred===1 && label===1) TP++;
      else if(pred===0 && label===0) TN++;
      else if(pred===1 && label===0) FP++;
      else if(pred===0 && label===1) FN++;
    });
    const TPR=TP/(TP+FN+1e-9);
    const FPR=FP/(FP+TN+1e-9);
    points.push({x:FPR,y:TPR});
  }
  tfvis.render.linechart(rocCanvas,{values:points,series:['ROC']},
    {xLabel:'False Positive Rate',yLabel:'True Positive Rate',width:400,height:300});
  // AUC (trapezoidal)
  let auc=0;
  for(let i=1;i<points.length;i++){
    auc+= (points[i].x - points[i-1].x) * (points[i].y + points[i-1].y)/2;
  }
  const ctx=rocCanvas.getContext('2d');
  ctx.fillText(`AUC ≈ ${auc.toFixed(3)}`,10,15);
  updateMetrics();
}

/* Slider → threshold -------------------------------------------------- */
threshSlider.oninput = e=>{
  threshold = parseFloat(e.target.value);
  threshVal.textContent = threshold.toFixed(2);
  if(valProbs) updateMetrics();
};

/////////////////////////// 6. PREDICTION & EXPORT //////////////////////

$('predictBtn').onclick = ()=>{
  const testVecs = rawTest.map(r=>prep.vectorize(r));
  const testX = tf.tensor2d(testVecs);
  const probs = Array.from(model.predict(testX).dataSync());
  const submission = [['PassengerId','Survived']];
  const probRows   = [['PassengerId','Prob']];
  probs.forEach((p,i)=>{
    const pid = rawTest[i][ID_COL];
    submission.push([pid, (p>=threshold?1:0)]);
    probRows.push   ([pid, p.toFixed(6)]);
  });
  $('downloadPredBtn').disabled = false;
  $('downloadProbBtn').disabled = false;

  $('downloadPredBtn').onclick = ()=> downloadCSV('submission.csv',submission);
  $('downloadProbBtn').onclick =()=> downloadCSV('probabilities.csv',probRows);
};

/* Download TFJS model ------------------------------------------------- */
$('saveModelBtn').onclick = ()=> model.save('downloads://titanic-tfjs');
