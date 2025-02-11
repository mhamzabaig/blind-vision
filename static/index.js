const pc = new RTCPeerConnection();

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });

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

document.getElementById("start").addEventListener("click", startVideo);
peerConnection.ontrack = (event) => {
  console.log("🔴 Receiving processed video stream");
  document.getElementById("outputVideo").srcObject = event.streams[0];
};
