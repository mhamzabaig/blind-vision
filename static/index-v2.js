const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});


async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  document.getElementById("stopButton").disabled = false;

  // ✅ Show the raw input video separately
  document.getElementById("inputVideo").srcObject = stream;

  // ✅ Send the input video to the server
  stream.getTracks().forEach((track) => pc.addTrack(track, stream));


  // ✅ Receive the processed output separately
  pc.ontrack = (event) => {
    console.log("✅ Processed video received from server");
    document.getElementById("outputVideo").srcObject = event.streams[0];
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const response = await fetch("/offer", {
    method: "POST",
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
    headers: { "Content-Type": "application/json" },
  });

  const answer = await response.json();
  await pc.setRemoteDescription(new RTCSessionDescription(answer));
}

document.getElementById("startButton").addEventListener("click", startVideo);
pc.ontrack = (event) => {
  console.log("🔴 Receiving processed video stream");
  document.getElementById("outputVideo").srcObject = event.streams[0];
};

document.getElementById("stopButton").addEventListener("click", stopVideo);
function stopVideo() {
  // ✅ Stop all media tracks
  const inputVideo = document.getElementById("inputVideo");
  if (inputVideo.srcObject) {
    inputVideo.srcObject.getTracks().forEach(track => track.stop());
    inputVideo.srcObject = null;
  }

  // ✅ Stop output video stream (processed video)
  const outputVideo = document.getElementById("outputVideo");
  if (outputVideo.srcObject) {
    outputVideo.srcObject.getTracks().forEach(track => track.stop());
    outputVideo.srcObject = null;
  }

  // ✅ Close WebRTC connection
  // pc.getSenders().forEach(sender => pc.removeTrack(sender));
  pc.close();

  console.log("🛑 Streaming stopped");

  // ✅ Disable stop button & re-enable start button
  document.getElementById("stopButton").disabled = true;
  document.getElementById("startButton").disabled = false;
}