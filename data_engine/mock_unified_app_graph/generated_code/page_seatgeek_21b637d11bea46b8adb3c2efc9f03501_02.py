# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_02
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5.png
# step_index: 2/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & UI structure drawing for 1440x2960 canvas
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_* (not used)

W, H = 1440, 2960

# Fill overall background with an off-white tone
draw.rectangle((0, 0, W, H), fill=(250, 250, 250))

# Status bar area (top subtle gray band)
status_h = 64
draw.rectangle((0, 0, W, status_h), fill=(240, 240, 240))

# Search bar background (rounded rect)
search_left, search_right = 48, W - 48
search_top, search_bottom = 48, 192  # centered around y ~120 where text is detected
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=28,
    fill=(245, 245, 245),
    outline=None
)

# Thin divider under the search area
divider_y = 240
draw.line((48, divider_y, W - 48, divider_y), fill=(230, 230, 230), width=2)

# "Recent searches" card background (large white rounded card)
card1_top = divider_y + 16
card1_bottom = 1360  # end of recent searches list area before suggestions
card_pad_left, card_pad_right = 48, W - 48
draw.rounded_rectangle(
    (card_pad_left, card1_top, card_pad_right, card1_bottom),
    radius=18,
    fill=(255, 255, 255),
    outline=(238, 238, 238),
    width=1
)

# Subtle horizontal separators inside the content area (not overlapping detected icons/text)
# These are light lines to indicate section breaks (kept minimal)
draw.line((card_pad_left + 8, card1_bottom, card_pad_right - 8, card1_bottom), fill=(235, 235, 235), width=1)

# "Suggestions" card background (another white rounded card)
suggest_top = 1424
suggest_bottom = 2000
draw.rounded_rectangle(
    (card_pad_left, suggest_top, card_pad_right, suggest_bottom),
    radius=18,
    fill=(255, 255, 255),
    outline=(238, 238, 238),
    width=1
)

# Divider above bottom navigation
nav_top = 2792
draw.line((0, nav_top, W, nav_top), fill=(230, 230, 230), width=2)

# Bottom navigation background (slightly elevated white strip)
draw.rectangle((0, nav_top, W, H), fill=(255, 255, 255))

# Subtle top shadow for the first card to give depth
shadow_t = card1_top - 8
draw.rectangle((card_pad_left, shadow_t, card_pad_right, card1_top), fill=(250, 250, 250))

# Small top divider between status bar and content area (subtle)
draw.line((0, status_h, W, status_h), fill=(230, 230, 230), width=1)

# Final subtle accents: very faint vertical gutters to mimic app margins
gutter_color = (248, 248, 248)
draw.rectangle((0, 0, 24, H), fill=gutter_color)
draw.rectangle((W - 24, 0, W, H), fill=gutter_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 69)
    canvas.paste(_c1, (1152, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1152, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 99, 69)
    canvas.paste(_c3, (1214, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1214, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/04_icon_NBA_Playoffs.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 807), _c4)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/05_icon_Just_Announced_by_My_Performers.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1688), _c5)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/07_icon_Drake.png
try:
    _c7 = get_crop(7, 1440, 168)
    canvas.paste(_c7, (0, 639), _c7)
except Exception:
    pass
layout["Drake"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/08_icon_Sofia_Isella.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 975), _c8)
except Exception:
    pass
layout["Sofia_Isella"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/10_icon_Clear.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 120), _c10)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 68)
    canvas.paste(_c11, (1319, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/12_icon_Events_by_My_Performers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1520), _c12)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/13_icon_NBA_Playoffs.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 639), _c13)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/14_icon_GK.png
try:
    _c14 = get_crop(14, 60, 62)
    canvas.paste(_c14, (178, 2), _c14)
except Exception:
    pass
layout["GK"] = [178, 2, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/15_icon_Account.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (1152, 2792), _c15)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/16_icon_Brooklyn_Nets.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 471), _c16)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/17_icon_6.37.png
try:
    _c17 = get_crop(17, 168, 144)
    canvas.paste(_c17, (48, 120), _c17)
except Exception:
    pass
layout["6.37"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/18_icon_Just_Announced_by_My_Performers.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1856), _c18)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/19_icon_Austin_FC.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 1143), _c19)
except Exception:
    pass
layout["Austin_FC"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/20_icon_6.37.png
try:
    _c20 = get_crop(20, 163, 64)
    canvas.paste(_c20, (8, 1), _c20)
except Exception:
    pass
layout["6.37"] = [8, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/23_text_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/24_text_Recent_searches.png
try:
    _c24 = get_crop(24, 168, 144)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_02_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-5/25_text_Suggestions.png
try:
    _c25 = get_crop(25, 331, 74)
    canvas.paste(_c25, (40, 1423), _c25)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
