# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_02
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5.png
# step_index: 2/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 250))

# Status bar background (top area)
status_bar_height = 120
draw.rectangle((0, 0, 1440, status_bar_height), fill=(242, 242, 242))
# subtle bottom divider for status bar
draw.line((0, status_bar_height, 1440, status_bar_height), fill=(225, 225, 225), width=1)

# Search bar background (rounded)
search_x0, search_y0 = 216, 120
search_x1, search_y1 = search_x0 + 1032, search_y0 + 144
draw.rounded_rectangle((search_x0, search_y0, search_x1, search_y1),
                       radius=28,
                       fill=(247, 247, 247),
                       outline=(230, 230, 230),
                       width=1)

# Divider under search area
divider_y = search_y1 + 0
draw.line((32, divider_y, 1440 - 32, divider_y), fill=(236, 236, 236), width=1)

# Recent searches card background
card1_x0, card1_y0 = 32, 312
card1_x1, card1_y1 = 1408, 1320
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=20,
                       fill=(255, 255, 255),
                       outline=(235, 235, 235),
                       width=1)

# Separators between list items inside card1 (approximate rows)
separators = [639, 807, 975, 1143, 1311]
for y in separators:
    # draw with a left inset so separators align like the UI (small left padding)
    draw.line((card1_x0 + 24, y, card1_x1 - 24, y), fill=(242, 242, 242), width=1)

# Divider between sections (visual separator above Suggestions)
draw.line((32, 1368, 1408, 1368), fill=(235, 235, 235), width=1)

# Suggestions card/background
card2_x0, card2_y0 = 32, 1408
card2_x1, card2_y1 = 1408, 2000
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1),
                       radius=16,
                       fill=(255, 255, 255),
                       outline=(235, 235, 235),
                       width=1)

# Subtle separators for suggestions (three items)
suggest_sep_positions = [1536 - 120, 1660 - 120]  # approximate internal separators
for y in suggest_sep_positions:
    draw.line((card2_x0 + 24, y, card2_x1 - 24, y), fill=(242, 242, 242), width=1)

# Bottom navigation bar background with top divider/shadow
nav_y0 = 2792
draw.rectangle((0, nav_y0, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_y0, 1440, nav_y0), fill=(230, 230, 230), width=2)

# Slight rounded highlight area for centered active area (background only)
active_indicator_w = 96
active_indicator_h = 8
active_x = (1440 // 2) - (active_indicator_w // 2)
active_y = nav_y0 + 18
draw.rounded_rectangle((active_x, active_y, active_x + active_indicator_w, active_y + active_indicator_h),
                       radius=4, fill=(255, 244, 240))

# Subtle overall vignette top/bottom lines to mimic real app separators
draw.line((32, card1_y0 - 24, 1408, card1_y0 - 24), fill=(245, 245, 245), width=1)
draw.line((32, card2_y1 + 8, 1408, card2_y1 + 8), fill=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/00_icon_New_York_Knicks.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/01_icon_New_York_Knicks.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 975), _c1)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/02_icon_Suggestions.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 1143), _c2)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 69)
    canvas.paste(_c3, (1152, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 64, 65)
    canvas.paste(_c4, (242, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/05_icon_Wembley_Stadium.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 471), _c5)
except Exception:
    pass
layout["Wembley_Stadium"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/06_icon_Tracking.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/08_icon_Browse.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/09_icon_Just_Announced_by_My_Performers.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 1688), _c9)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/11_icon_7_06_my.png
try:
    _c11 = get_crop(11, 168, 144)
    canvas.paste(_c11, (48, 120), _c11)
except Exception:
    pass
layout["7:06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 68)
    canvas.paste(_c13, (1319, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/14_icon_7_06_my.png
try:
    _c14 = get_crop(14, 47, 64)
    canvas.paste(_c14, (186, 1), _c14)
except Exception:
    pass
layout["7:06_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/15_icon_Golden_State_Warriors.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 807), _c15)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 62, 64)
    canvas.paste(_c16, (313, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/18_icon_Golden_State_Warriors.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 975), _c18)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/19_icon_Cryptocom_Arena.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 639), _c19)
except Exception:
    pass
layout["Cryptocom_Arena"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/20_icon_Events_by_My_Performers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1520), _c20)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/21_icon_7_06_my.png
try:
    _c21 = get_crop(21, 58, 65)
    canvas.paste(_c21, (113, 0), _c21)
except Exception:
    pass
layout["7:06_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/23_icon_Los_Angeles_Clippers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_02_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-5/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
