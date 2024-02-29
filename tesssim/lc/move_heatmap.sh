hm = "amp_mag_heatmap.png"
for d in */ ; do
    cp "${d}amp_mag_heatmap.png" "amp_mag_heatmap_${d%/}.png"
done

