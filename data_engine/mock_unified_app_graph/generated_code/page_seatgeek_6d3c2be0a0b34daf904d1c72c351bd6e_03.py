# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_03
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6.png
# step_index: 3/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background canvas is provided as `canvas` (1440x2960 RGB) and `draw` (ImageDraw).
# Fonts available: font_sm, font_md, font_lg, font_xl

# Fill overall background (reinforce white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(245, 245, 245))
# subtle bottom hairline under status bar
draw.line([(0, status_h-1), (1440, status_h-1)], fill=(225, 225, 225), width=1)

# Search input background (rounded) - placed behind detected search widgets
search_left = 48
search_top = 96
search_right = 1392  # 1440 - 48
search_bottom = search_top + 144
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=20,
    fill=(250, 250, 250),
    outline=(230, 230, 230),
    width=1
)

# Light divider under the header/search area
divider_y = search_bottom + 20
draw.line([(48, divider_y), (1392, divider_y)], fill=(235, 235, 235), width=2)

# Recent searches / list separators
# Detected list item rows start around y positions; draw subtle separators between rows
list_tops = [471, 639, 807, 975, 1143]  # top positions of recent search rows (detected)
for top in list_tops:
    bottom = top + 168
    # draw a thin separator line at the bottom of each row (light)
    draw.line([(48, bottom), (1392, bottom)], fill=(240, 240, 240), width=1)

# Stronger section divider between "Recent searches" and "Suggestions"
# estimated separator around where recent list ends (~1143 + 168 = 1311)
section_div_y = 1312
draw.line([(48, section_div_y), (1392, section_div_y)], fill=(230, 230, 230), width=2)

# Suggestions section separators (three suggestion rows)
suggestion_tops = [1520, 1688, 1856]
for top in suggestion_tops:
    bottom = top + 168
    draw.line([(48, bottom), (1392, bottom)], fill=(240, 240, 240), width=1)

# Bottom navigation bar area
nav_top = 2792
nav_bottom = 2960
# subtle top hairline/shadow for nav
draw.rectangle([(0, nav_top-8), (1440, nav_top)], fill=(245, 245, 245))
# nav background (keep white to match screenshot)
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))
# thin divider above nav
draw.line([(48, nav_top), (1392, nav_top)], fill=(230, 230, 230), width=1)

# Optional subtle left/right margins guide lines (very light) to anchor content areas
draw.line([(48, status_h + 40), (48, nav_top - 40)], fill=(250, 250, 250), width=1)
draw.line([(1392, status_h + 40), (1392, nav_top - 40)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/00_icon_New_York_Knicks.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 975), _c0)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/01_icon_New_York_Knicks.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 807), _c1)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/02_icon_Suggestions.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 1143), _c2)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/03_icon_Wembley_Stadium.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["Wembley_Stadium"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 47, 70)
    canvas.paste(_c4, (1153, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/05_icon_7_06_my.png
try:
    _c5 = get_crop(5, 168, 144)
    canvas.paste(_c5, (48, 120), _c5)
except Exception:
    pass
layout["7:06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/06_icon_Tracking.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 64)
    canvas.paste(_c8, (242, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [242, 2, 305, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 96, 69)
    canvas.paste(_c10, (1216, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/12_icon_Golden_State_Warriors.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/14_icon_Cryptocom_Arena.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 639), _c14)
except Exception:
    pass
layout["Cryptocom_Arena"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/15_icon_Golden_State_Warriors.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 807), _c15)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 68)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/17_icon_Events_by_My_Performers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1520), _c17)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 60, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/20_icon_7_06_my.png
try:
    _c20 = get_crop(20, 47, 64)
    canvas.paste(_c20, (186, 1), _c20)
except Exception:
    pass
layout["7:06_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/22_icon_Performer_event_or_venue.png
try:
    _c22 = get_crop(22, 1032, 144)
    canvas.paste(_c22, (216, 120), _c22)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/23_icon_Los_Angeles_Clippers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/26_text_7_06_my.png
try:
    _c26 = get_crop(26, 156, 52)
    canvas.paste(_c26, (19, 9), _c26)
except Exception:
    pass
layout["7:06_my"] = [19, 9, 175, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_03_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
