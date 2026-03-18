# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_05
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7.png
# step_index: 5/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas using provided `canvas` and `draw`
# Variables available: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Ensure full white background (canvas is already white, but redraw to be explicit)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# ---------- Status bar ----------
# Light gray status bar at top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#D9D9D9")

# Thin separator under status bar (subtle)
draw.line([(0, status_h), (1440, status_h)], fill="#CFCFCF", width=1)

# ---------- Header / Toolbar ----------
# Header area (keeps white background but we draw to ensure consistent look)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Prominent blue underline for header (matching app accent)
underline_y = header_bottom
draw.line([(48, underline_y), (1392, underline_y)], fill="#2B56F5", width=6)

# Subtle top edge highlight for header
draw.line([(0, header_top), (1440, header_top)], fill="#F6F6F6", width=1)

# ---------- Nearby item card (group background) ----------
# A subtle rounded container behind the "Nearby"/current location area
nearby_card_top = 200
nearby_card_bottom = 360
left_margin = 40
right_margin = 1400
card_outline = "#E8F0FF"
card_fill = "#FFFFFF"  # keep white so it doesn't duplicate text/icons, just gives subtle framed area
draw.rounded_rectangle(
    [(left_margin, nearby_card_top), (right_margin, nearby_card_bottom)],
    radius=12,
    fill=card_fill,
    outline=card_outline,
    width=1
)

# ---------- Thin divider between header/nearby and the list ----------
divider_y = nearby_card_bottom + 40
draw.line([(48, divider_y), (1392, divider_y)], fill="#EEF1F6", width=1)

# ---------- Found locations list background ----------
# Slightly off-white background band for the long list area (keeps contrast but doesn't cover text)
list_top = 720
list_bottom = 2600
list_left = 0
list_right = 1440
# Use a very subtle tint so pasted text/icons remain unchanged in appearance
draw.rectangle([(list_left, list_top), (list_right, list_bottom)], fill="#FFFFFF")

# Add a very faint left padding guideline (visual structure only, not text)
draw.line([(48, list_top), (48, list_bottom)], fill="#F3F4F6", width=1)

# ---------- Separators between list items ----------
# Based on detected rows, draw subtle separators between each location row.
separator_color = "#F2F3F6"
# List rows roughly every 180 px starting at 840 (as seen in detections)
row_starts = [972, 1152, 1332, 1512, 1692, 1872, 2052, 2232, 2412]
for y in row_starts:
    draw.line([(48, y), (1392, y)], fill=separator_color, width=1)

# ---------- Bottom padding line ----------
draw.line([(0, list_bottom), (1440, list_bottom)], fill="#F6F6F8", width=1)

# ---------- Floating subtle accents (do not draw icons/text) ----------
# A faint vertical accent near the header left (purely decorative, avoids icon/text areas)
accent_x = 40
draw.line([(accent_x, header_top + 20), (accent_x, underline_y - 6)], fill="#E9F0FF", width=6)

# End of background/structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/02_icon_5.34.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["5.34"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/03_icon_5.34.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["5.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/04_icon_5.34.png
try:
    _c4 = get_crop(4, 62, 66)
    canvas.paste(_c4, (113, 0), _c4)
except Exception:
    pass
layout["5.34"] = [113, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 61)
    canvas.paste(_c5, (308, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [308, 2, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 85, 91)
    canvas.paste(_c6, (1310, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1310, 288, 1395, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 62)
    canvas.paste(_c7, (246, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [246, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 58)
    canvas.paste(_c8, (1322, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/10_icon_Chicago.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1380), _c10)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/11_icon_Los_Angeles.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1020), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/12_icon_Miami.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1200), _c12)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/13_icon_5.34.png
try:
    _c13 = get_crop(13, 94, 63)
    canvas.paste(_c13, (14, 2), _c13)
except Exception:
    pass
layout["5.34"] = [14, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/14_icon_District_of_Columbia.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1560), _c14)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1740), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1344, 129)
    canvas.paste(_c16, (48, 264), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/17_text_Nearby.png
try:
    _c17 = get_crop(17, 415, 114)
    canvas.paste(_c17, (48, 465), _c17)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/18_text_Current_location.png
try:
    _c18 = get_crop(18, 415, 114)
    canvas.paste(_c18, (48, 465), _c18)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/19_text_Found_locations.png
try:
    _c19 = get_crop(19, 311, 50)
    canvas.paste(_c19, (44, 740), _c19)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/20_text_Philadelphia.png
try:
    _c20 = get_crop(20, 1440, 132)
    canvas.paste(_c20, (0, 1920), _c20)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/21_text_Pennsylvania.png
try:
    _c21 = get_crop(21, 214, 43)
    canvas.paste(_c21, (45, 1995), _c21)
except Exception:
    pass
layout["Pennsylvania"] = [45, 1995, 259, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/22_text_London.png
try:
    _c22 = get_crop(22, 168, 52)
    canvas.paste(_c22, (44, 2109), _c22)
except Exception:
    pass
layout["London"] = [44, 2109, 212, 2161]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/23_text_United_Kingdom.png
try:
    _c23 = get_crop(23, 263, 45)
    canvas.paste(_c23, (45, 2173), _c23)
except Exception:
    pass
layout["United_Kingdom"] = [45, 2173, 308, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/24_text_New_York.png
try:
    _c24 = get_crop(24, 212, 55)
    canvas.paste(_c24, (44, 2288), _c24)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 154, 38)
    canvas.paste(_c25, (47, 2353), _c25)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/26_text_Atlanta.png
try:
    _c26 = get_crop(26, 163, 52)
    canvas.paste(_c26, (44, 2468), _c26)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/27_text_Georgia.png
try:
    _c27 = get_crop(27, 133, 43)
    canvas.paste(_c27, (45, 2533), _c27)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/28_clickable_London.png
try:
    _c28 = get_crop(28, 1440, 132)
    canvas.paste(_c28, (0, 2100), _c28)
except Exception:
    pass
layout["London"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_05_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-7/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
