let model;
window.addEventListener('load', async () => {
    model = await tf.loadGraphModel('./model-js/model.json');
})

function onImageChange(event) {
    const imagePath = URL.createObjectURL(event.target.files[0]);
    predict(imagePath);
}

function predict(imagePath) {
    labels = ['No plane or Ship', 'Plane', 'Ship'];
    const img = new Image();
    img.src = imagePath;

    img.onload = async () => {
        const a = tf.browser.fromPixels(img, 3).toFloat();
        const prediction = await model.predict(a.expandDims(0)).data();
        const correctClass = Array.from(prediction).map((prob, i) => {
            return {
                class: labels[i],
                probability: prob
            }
        }).sort((a, b) => {
            return b.probability - a.probability;
        });
        document.getElementById('predicted-class').innerHTML = correctClass[0].class;
        document.getElementById('probability').innerHTML = correctClass[0].probability;
    }
}
