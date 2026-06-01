document.addEventListener("DOMContentLoaded", () => {
  const detectBtn = document.getElementById("detectBtn");
  const brandName = document.getElementById("brandName");
  const shadeValues = document.getElementById("shadeValues");
  const shadePalette = document.getElementById("shadePalette");

  if (!detectBtn) {
    console.error("❌ detectBtn not found in DOM!");
    return;
  }

  detectBtn.addEventListener("click", async () => {
    brandName.textContent = "Detecting your perfect shade... 💫";
    shadeValues.textContent = "";
    shadePalette.style.display = "none";

    try {
      const response = await fetch("/detect");
      const data = await response.json();

      if (response.ok && data.status === "success") {
        brandName.textContent = `✨ ${data.brand.toUpperCase()} ✨`;
        shadeValues.innerHTML = `
          <b>Shade Palette (RGB):</b><br>
          ${JSON.stringify(data.matched_shade)}
        `;
        if (data.matched_image) {
          shadePalette.src = data.matched_image;
          shadePalette.style.display = "inline-block";
        }
      } else if (data.error) {
        brandName.textContent = "⚠️ Error detecting shade!";
        shadeValues.innerHTML = `<small>${data.error}</small>`;
      } else {
        brandName.textContent = "⚠️ Unknown response from server.";
      }
    } catch (err) {
      console.error(err);
      brandName.textContent = "❌ Could not connect to Flask backend.";
    }
  });
});
