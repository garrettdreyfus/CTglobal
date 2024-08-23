function normalize(val, min, max){
  // Shift to positive to avoid issues when crossing the 0 line
  if(min < 0){
    max += 0 - min;
    val += 0 - min;
    min = 0;
  }
  // Shift values from 0 - max
  val = val - min;
  max = max - min;
  return Math.max(0, Math.min(1, val / max));
}

function getMax(a){
  return Math.max(...a.flat());
}

function getMin(a){
  return Math.min(...a.flat());
}

function terrainA(a){
        out = ocean_r(a)
        out.push(255)
        return out
}

srtm = srtm.reverse()
ilen = srtm.length;
jlen = srtm[0].length;

const MAP = document.createElement("canvas");
const MAPctx = MAP.getContext("2d");


const OUT = document.createElement("canvas");
const OUTctx = OUT.getContext("2d");

function getCursorPosition(canvas, evt) {
    var rect = canvas.getBoundingClientRect(), // abs. size of element
    scaleX = canvas.width / rect.width,    // relationship bitmap vs. element for x
    scaleY = canvas.height / rect.height;  // relationship bitmap vs. element for y
  return {
    x: (evt.clientX - rect.left) * scaleX,   // scale mouse coordinates after they have
    y: (evt.clientY - rect.top) * scaleY     // been adjusted to be relative to element
  }
}

selectionbuffer = [[],[]];

MAP.addEventListener('mousedown', function(e) {
    const out = getCursorPosition(MAP, e);
    if (selectionbuffer[0].length==2) {
        selectionbuffer = [[],[]];
        drawMap();
    }
    else if (selectionbuffer[0].length<2) {
        console.log(out)
        selectionbuffer[0].push(out.x);
        selectionbuffer[1].push(out.y);
        x1 = Math.round(out.x)
        y1 = Math.round(out.y)
        MAPctx.fillStyle = "#FF0000";
        MAPctx.beginPath();
        MAPctx.arc(out.x,out.y, 5, 0, 2*Math.PI);
        MAPctx.fill();
    }
    if (selectionbuffer[0].length==2) {
        findConnection(selectionbuffer)
    }
})

function drawMap() {
 
    //srtm=srtm.map(row=>row.reverse()).reverse()

    srtmin = getMin(srtm)
    const flattenedRGBAValues = srtm
      .flat()  // 1d list of ints codes
      .map(e => terrainA((e)/srtmin))  // 1d list of [R, G, B, A] byte arrays
      .flat(); // 1d list of bytes
               ////
    // Render on screen for demo
    MAP.width = jlen;
    MAP.height = jlen;
    MAPctx.clearRect(0, 0, MAP.width, MAP.height);
    const imgData = new ImageData(Uint8ClampedArray.from(flattenedRGBAValues), jlen, ilen);
    MAPctx.putImageData(imgData, 0, 0);

}


window.onload = function() {
    document.body.appendChild(MAP);
    document.body.appendChild(OUT);
    drawMap();
}
// slice prototype
const delay = ms => new Promise(res => setTimeout(res, ms));
function findConnection() {
    console.log(selectionbuffer)
    x1 = Math.round(selectionbuffer[0][0]);
    x2 = Math.round(selectionbuffer[0][1]);
    y1 = Math.round(selectionbuffer[1][0]);
    y2 = Math.round(selectionbuffer[1][1]);

    z1 = srtm[y1][x1];
    z2 = srtm[y2][x2];

    console.log(z1,z2)


    for (let thresh = Math.max(z1,z2); thresh < 0; thresh+=100){
        mask = srtm.map(e=> e.map(j=>(Number(j>thresh))*256+-1));
        blobs = BlobExtraction(mask.flat(), jlen, ilen)
        blobmax = getMax(blobs)

       // Render on screen for demo
       if(blobs[x1+y1*jlen] == blobs[x2+jlen*y2]){
            OUT.width = jlen;
            OUT.height = jlen;
            OUTctx.clearRect(0, 0, OUT.width, OUT.height);
	    blobs = blobs.map(e => Number(e==blobs[x1+y1*jlen]))
	    const flattenedRGBAValues = blobs
	      .map(terrainA)  // 1d list of [R, G, B, A] byte arrays
	      .flat(); // 1d list of bytes

     
            const imgData = new ImageData(Uint8ClampedArray.from(flattenedRGBAValues), jlen, ilen);
            OUTctx.putImageData(imgData, 0, 0);

	    OUTctx.fillStyle = "#000000";
	    OUTctx.beginPath();
            OUTctx.arc(x1,y1, 5, 0, 2*Math.PI);
            OUTctx.arc(x2,y2, 5, 0, 2*Math.PI);
	    OUTctx.fill();
            console.log(srtm.flat())
	    elem = document.getElementById("depth")

	    elem.innerHTML = thresh;
                break;
        }
    }

    
}
