# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_11
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13.png
# step_index: 11/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar area at top (~56px) - subtle light gray
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#d9d9d9")

# Header/banner area behind the top image (purple gradient)
header_top = status_h
header_bottom = 360
# Simple vertical gradient for header
start_rgb = (59, 15, 87)   # deep purple
end_rgb = (103, 41, 115)   # slightly lighter purple
h = header_bottom - header_top
for i in range(h):
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / max(h - 1, 1)))
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / max(h - 1, 1)))
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / max(h - 1, 1)))
    draw.line([(0, header_top + i), (1440, header_top + i)], fill=(r, g, b))

# Slight darker overlay band near bottom of header to mimic vignette/shadow
overlay_top = header_bottom - 36
for i in range(36):
    alpha = int(20 * (i / 35.0))  # subtle darkening
    y = overlay_top + i
    draw.line([(0, y), (1440, y)], fill=(0, 0, 0, alpha))

# Main content white rounded card area (content panel)
content_top = 320
content_bottom = 2320  # stop above the reserve area (reserve area begins at y=2324)
content_margin = 24
draw.rounded_rectangle(
    [(content_margin, content_top), (1440 - content_margin, content_bottom)],
    radius=28,
    fill="#ffffff",
    outline="#e9e9ee",
    width=1
)

# Organizer / host card background (light neutral rounded rectangle)
# Positioned roughly where the organizer block sits; keep it only as a background plate.
org_top = 1068
org_bottom = 1244
org_left = 36
org_right = 1404
draw.rounded_rectangle(
    [(org_left, org_top), (org_right, org_bottom)],
    radius=20,
    fill="#f6f7fb",
    outline="#ececf3",
    width=2
)

# Add subtle inner shadow to organizer card (top and bottom thin strokes)
draw.line([(org_left + 2, org_top + 2), (org_right - 2, org_top + 2)], fill="#f0f1f6", width=1)
draw.line([(org_left + 2, org_bottom - 2), (org_right - 2, org_bottom - 2)], fill="#efeff4", width=1)

# Thin divider/separator under the informational list (approx area of refund policy divider)
sep_y1 = 1420
draw.line([(48, sep_y1), (1440 - 48, sep_y1)], fill="#e6e6ea", width=2)

# Secondary separator for grouping content sections
sep_y2 = 1600
draw.line([(48, sep_y2), (1440 - 48, sep_y2)], fill="#f0f0f3", width=1)

# Section heading accent (visual underline bar for "Select date and time")
# Draw a small accent bar to the left of where the heading would be (no text)
accent_x = 48
accent_y = 1740
draw.rectangle([(accent_x, accent_y), (accent_x + 8, accent_y + 28)], fill="#3b0f57")

# Light background behind the date selection row (keeps visual grouping, but avoid drawing date cards)
dates_bg_top = 1840
dates_bg_bottom = 2020
draw.rectangle([(48, dates_bg_top), (1440 - 48, dates_bg_bottom)], fill="#ffffff", outline="#f0f0f3", width=1)

# Subtle drop shadow under the header image area (above content)
shadow_top = content_top - 6
for i in range(6):
    alpha = int(30 * (1 - i / 6.0))
    y = shadow_top + i
    draw.line([(48, y), (1440 - 48, y)], fill=(0, 0, 0, alpha))

# Left gutter vertical guide (thin decorative line to separate content column)
draw.line([(48, content_top + 36), (48, sep_y2 - 16)], fill="#f2f2f5", width=2)

# A faint rounded outline box where ticket selection would appear (but keep it above the reserve zone)
ticket_box_top = 2040
ticket_box_bottom = 2308  # keep entirely above reserve area start (2324)
draw.rounded_rectangle(
    [(48, ticket_box_top), (1392, ticket_box_bottom)],
    radius=18,
    fill="#ffffff",
    outline="#e8e9ef",
    width=3
)

# Inner subtle horizontal separator inside ticket box
draw.line([(72, ticket_box_top + 64), (1368, ticket_box_top + 64)], fill="#f0f0f3", width=1)

# End of structural drawing. (No icons, texts or buttons were drawn to avoid duplicating detected elements.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/00_icon_May.png
try:
    _c0 = get_crop(0, 450, 257)
    canvas.paste(_c0, (924, 2067), _c0)
except Exception:
    pass
layout["May"] = [924, 2067, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/01_icon_Follow.png
try:
    _c1 = get_crop(1, 331, 144)
    canvas.paste(_c1, (1013, 1163), _c1)
except Exception:
    pass
layout["Follow"] = [1013, 1163, 1344, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/02_icon_April.png
try:
    _c2 = get_crop(2, 450, 257)
    canvas.paste(_c2, (24, 2067), _c2)
except Exception:
    pass
layout["April"] = [24, 2067, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/03_icon_May.png
try:
    _c3 = get_crop(3, 450, 257)
    canvas.paste(_c3, (474, 2067), _c3)
except Exception:
    pass
layout["May"] = [474, 2067, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/04_icon_May.png
try:
    _c4 = get_crop(4, 108, 104)
    canvas.paste(_c4, (989, 2440), _c4)
except Exception:
    pass
layout["May"] = [989, 2440, 1097, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1440, 636)
    canvas.paste(_c5, (0, 2324), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 104, 102)
    canvas.paste(_c6, (1218, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1218, 2441, 1322, 2543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/07_icon_May.png
try:
    _c7 = get_crop(7, 93, 101)
    canvas.paste(_c7, (1108, 2442), _c7)
except Exception:
    pass
layout["May"] = [1108, 2442, 1201, 2543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/08_icon_5.25.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["5.25"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/09_icon_INTERACTIVE.png
try:
    _c9 = get_crop(9, 60, 67)
    canvas.paste(_c9, (181, 1), _c9)
except Exception:
    pass
layout["INTERACTIVE"] = [181, 1, 241, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/10_icon_PRESENTER.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1260, 108), _c10)
except Exception:
    pass
layout["PRESENTER"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/11_icon_5.25.png
try:
    _c11 = get_crop(11, 57, 66)
    canvas.paste(_c11, (116, 1), _c11)
except Exception:
    pass
layout["5.25"] = [116, 1, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/12_icon_INTERACTIVE.png
try:
    _c12 = get_crop(12, 51, 62)
    canvas.paste(_c12, (248, 4), _c12)
except Exception:
    pass
layout["INTERACTIVE"] = [248, 4, 299, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/13_icon_PRESENTER.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["PRESENTER"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/14_icon_INTERACTIVE.png
try:
    _c14 = get_crop(14, 67, 64)
    canvas.paste(_c14, (307, 3), _c14)
except Exception:
    pass
layout["INTERACTIVE"] = [307, 3, 374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/15_icon_Free.png
try:
    _c15 = get_crop(15, 136, 112)
    canvas.paste(_c15, (98, 2570), _c15)
except Exception:
    pass
layout["Free"] = [98, 2570, 234, 2682]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 48, 59)
    canvas.paste(_c16, (1266, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [1266, 2, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 64, 59)
    canvas.paste(_c17, (1217, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1217, 2, 1281, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/18_icon_LIVE_Q_A.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1116, 108), _c18)
except Exception:
    pass
layout["LIVE_Q&A"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 44, 58)
    canvas.paste(_c19, (1327, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [1327, 4, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/20_icon_Free.png
try:
    _c20 = get_crop(20, 100, 128)
    canvas.paste(_c20, (232, 2565), _c20)
except Exception:
    pass
layout["Free"] = [232, 2565, 332, 2693]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 49, 64)
    canvas.paste(_c21, (383, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [383, 3, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/22_icon_EVERYTHING_YOU.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (36, 108), _c22)
except Exception:
    pass
layout["EVERYTHING_YOU"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/23_text_5.25.png
try:
    _c23 = get_crop(23, 95, 50)
    canvas.paste(_c23, (20, 13), _c23)
except Exception:
    pass
layout["5.25"] = [20, 13, 115, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/24_text_Thursday_April_25.png
try:
    _c24 = get_crop(24, 456, 77)
    canvas.paste(_c24, (40, 758), _c24)
except Exception:
    pass
layout["Thursday_April_25"] = [40, 758, 496, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/25_text_IO_O0_AM.png
try:
    _c25 = get_crop(25, 242, 54)
    canvas.paste(_c25, (527, 768), _c25)
except Exception:
    pass
layout["IO:O0_AM"] = [527, 768, 769, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/26_text_Everything_You_Need_To_Know_About.png
try:
    _c26 = get_crop(26, 394, 144)
    canvas.paste(_c26, (288, 1123), _c26)
except Exception:
    pass
layout["Everything_You_Need_To_Kn"] = [288, 1123, 682, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/27_text_Mytel.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (96, 1162), _c27)
except Exception:
    pass
layout["Mytel"] = [96, 1162, 240, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/28_text_My_Tech_Academy.png
try:
    _c28 = get_crop(28, 394, 144)
    canvas.paste(_c28, (288, 1123), _c28)
except Exception:
    pass
layout["My_Tech_Academy"] = [288, 1123, 682, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/29_text_1.S5k_Followers.png
try:
    _c29 = get_crop(29, 394, 144)
    canvas.paste(_c29, (288, 1123), _c29)
except Exception:
    pass
layout["1.S5k_Followers"] = [288, 1123, 682, 1267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/30_text_Online_event.png
try:
    _c30 = get_crop(30, 274, 54)
    canvas.paste(_c30, (139, 1436), _c30)
except Exception:
    pass
layout["Online_event"] = [139, 1436, 413, 1490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/31_text_hrs.png
try:
    _c31 = get_crop(31, 77, 50)
    canvas.paste(_c31, (176, 1547), _c31)
except Exception:
    pass
layout["hrs"] = [176, 1547, 253, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/32_text_Refund_policy.png
try:
    _c32 = get_crop(32, 299, 63)
    canvas.paste(_c32, (138, 1653), _c32)
except Exception:
    pass
layout["Refund_policy"] = [138, 1653, 437, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/33_text_The_organizer_will_review_refund_request.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1390), _c33)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1390, 1392, 1534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/34_text_Select_date_and_time.png
try:
    _c34 = get_crop(34, 450, 257)
    canvas.paste(_c34, (24, 2067), _c34)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2067, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_11_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-13/35_text_General_Admission.png
try:
    _c35 = get_crop(35, 450, 257)
    canvas.paste(_c35, (24, 2067), _c35)
except Exception:
    pass
layout["General_Admission"] = [24, 2067, 474, 2324]
