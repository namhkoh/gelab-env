# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_02
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5.png
# step_index: 2/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for provided canvas/draw objects.
# Assumes: canvas (1440x2960 RGB PIL Image) and draw (ImageDraw) are already defined.

w, h = canvas.size

# Colors
bg_color = (250, 250, 250)         # page background (very light off-white)
status_bar_color = (243, 244, 245) # subtle top status area
search_bg = (242, 243, 244)        # search bar background
search_border = (230, 230, 230)    # subtle border/shadow for search bar
divider_color = (230, 230, 230)    # section dividers
nav_border = (232, 232, 232)       # top border of bottom nav
nav_bg = (255, 255, 255)           # bottom nav background (white)

# Clear whole canvas to the page background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (approx height 80px)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Search bar (rounded rectangle)
search_left = 48
search_right = w - 48
search_top = 72
search_bottom = 216
search_radius = 28
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_bg,
    outline=search_border,
    width=1
)

# Subtle line/shadow directly under search bar to separate from content
shadow_top = search_bottom + 4
shadow_bottom = shadow_top + 2
draw.rectangle([(search_left, shadow_top), (search_right, shadow_bottom)], fill=search_border)

# Full-width thin divider under header area
divider_y = search_bottom + 24
draw.line([(32, divider_y), (w - 32, divider_y)], fill=divider_color, width=1)

# Separator between "Recent searches" list and "Suggestions" (approx based on detected positions)
# Using the detected Suggestions text Y ~1423, draw a subtle divider there
suggestions_div_y = 1424
draw.line([(32, suggestions_div_y), (w - 32, suggestions_div_y)], fill=divider_color, width=1)

# Slight inset horizontal divider under the recent searches header area (to echo screenshot)
small_div_y = 480  # a light divider between top list groups
draw.line([(48, small_div_y), (w - 48, small_div_y)], fill=(240,240,240), width=1)

# Bottom navigation bar background and top border
nav_top = 2792
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
draw.line([(0, nav_top), (w, nav_top)], fill=nav_border, width=1)

# Optional: faint large content area band to hint image/content regions (keeps clear of detected elements)
# This is a very subtle band behind where content cards/images would appear (not drawing any icons/text)
content_band_top = 920
content_band_bottom = 1160
content_band_inset = 40
draw.rectangle(
    [(content_band_inset, content_band_top), (w - content_band_inset, content_band_bottom)],
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)

# Another subtle content band for lower announcement cards (keeps clear of detected elements)
lower_band_top = 1500
lower_band_bottom = 1920
draw.rectangle(
    [(content_band_inset, lower_band_top), (w - content_band_inset, lower_band_bottom)],
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/01_icon_Brooklyn_Nets.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 51, 69)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 99, 68)
    canvas.paste(_c3, (1214, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1214, 0, 1313, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/06_icon_GEK.png
try:
    _c6 = get_crop(6, 62, 62)
    canvas.paste(_c6, (245, 2), _c6)
except Exception:
    pass
layout["GEK"] = [245, 2, 307, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/07_icon_Clear.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 120), _c7)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/08_icon_7.40_Wy.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["7.40_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 67)
    canvas.paste(_c9, (1319, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 0, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (576, 2792), _c11)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/12_icon_NBA_Playoffs.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/13_icon_NBA_Playoffs.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/16_icon_GEK.png
try:
    _c16 = get_crop(16, 64, 63)
    canvas.paste(_c16, (176, 0), _c16)
except Exception:
    pass
layout["GEK"] = [176, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/17_icon_Drake.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 807), _c17)
except Exception:
    pass
layout["Drake"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/18_icon_7.40_Wy.png
try:
    _c18 = get_crop(18, 161, 65)
    canvas.paste(_c18, (10, 0), _c18)
except Exception:
    pass
layout["7.40_Wy"] = [10, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/19_icon_Brooklyn_Nets.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 639), _c19)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/20_icon_Search.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/21_icon_Brooklyn_Nets.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 471), _c21)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/22_icon_Sofia_Isella.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Sofia_Isella"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/23_icon_Just_Announced_by_My_Performers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1856), _c23)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/24_text_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/25_text_Recent_searches.png
try:
    _c25 = get_crop(25, 168, 144)
    canvas.paste(_c25, (48, 120), _c25)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_02_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-5/26_text_Suggestions.png
try:
    _c26 = get_crop(26, 331, 74)
    canvas.paste(_c26, (40, 1423), _c26)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
