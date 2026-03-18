# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_05
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8.png
# step_index: 5/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level UI background and structure for the SeatGeek venue page

# Overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FAFAFB")

# Status bar area (approx ~72px high)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#ECEFF1")
# thin bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#D8D8D8", width=1)

# Map / hero background area (clickable map region starts at y=status_h with height 704)
map_y0 = status_h
map_h = 704
map_y1 = map_y0 + map_h
# subtle two-tone map background (light water / light land)
draw.rectangle([(0, map_y0), (1440, map_y1)], fill="#F0F6FB")
# horizontal subtle gradient band to suggest roads (simple stripes)
for i in range(6):
    y = map_y0 + int(map_h * (i+0.05) / 6)
    draw.line([(0, y), (1440, y)], fill="#F8FAFB", width=2)

# Card containing venue title (overlapping the map slightly) with shadow
card_margin = 24
venue_card_top = map_y1 - 56  # overlap to match screenshot feel
venue_card_bottom = venue_card_top + 320
venue_card_rect = (card_margin, venue_card_top, 1440 - card_margin, venue_card_bottom)
# shadow (simple darker rect offset)
shadow_offset = 6
draw.rounded_rectangle(
    [venue_card_rect[0] + shadow_offset, venue_card_rect[1] + shadow_offset,
     venue_card_rect[2] + shadow_offset, venue_card_rect[3] + shadow_offset],
    radius=20, fill="#E9EBEE"
)
# main card
draw.rounded_rectangle(venue_card_rect, radius=20, fill="#FFFFFF")
# subtle divider line inside venue card (below title area)
divider_y = venue_card_top + 140
draw.line([(venue_card_rect[0]+28, divider_y), (venue_card_rect[2]-28, divider_y)], fill="#E6E6E6", width=1)

# Popular events section container (card-like area for the horizontal scroller)
popular_top = divider_y + 36
popular_bottom = popular_top + 520
popular_rect = (card_margin, popular_top, 1440 - card_margin, popular_bottom)
draw.rectangle(popular_rect, fill="#FFFFFF")
# heading area divider (top of popular content)
draw.line([(popular_rect[0]+24, popular_top+64), (popular_rect[2]-24, popular_top+64)], fill="#FFFFFF", width=1)
# subtle bottom separator for the popular section
draw.line([(popular_rect[0], popular_bottom), (popular_rect[2], popular_bottom)], fill="#EFEFEF", width=1)

# Thumbnail/card backgrounds for the three popular event tiles.
# These are intentionally drawn as rounded background placeholders behind the actual detected thumbnails.
pop_tile_y = 1273  # as per detected icon y
pop_tile_h = 519   # approximate height for the first; other sizes vary slightly per detection
# First tile (left)
tile1 = (48, pop_tile_y, 48 + 462, pop_tile_y + 519)
draw.rounded_rectangle(tile1, radius=18, fill="#F6F6F8")
# Accent bottom overlay bar for price area (left)
draw.rectangle([(tile1[0], tile1[3] - 72), (tile1[2], tile1[3])], fill="#00000080")

# Second tile (middle)
tile2 = (546, 1273, 546 + 462, 1273 + 533)
draw.rounded_rectangle(tile2, radius=18, fill="#F6F6F8")
draw.rectangle([(tile2[0], tile2[3] - 72), (tile2[2], tile2[3])], fill="#00000080")

# Third tile (right)
tile3 = (1044, 1273, 1044 + 396, 1273 + 533)
draw.rounded_rectangle(tile3, radius=18, fill="#F6F6F8")
draw.rectangle([(tile3[0], tile3[3] - 72), (tile3[2], tile3[3])], fill="#00000080")

# Horizontal separator below popular events
sep_y = popular_bottom + 24
draw.line([(24, sep_y), (1440 - 24, sep_y)], fill="#E6E6E6", width=1)

# Seating charts section container
seating_top = sep_y + 40
seating_bottom = seating_top + 420
seating_rect = (24, seating_top, 1440 - 24, seating_bottom)
draw.rectangle(seating_rect, fill="#FFFFFF")
# small internal padding line (visual separator under the "Seating charts" header area)
draw.line([(seating_rect[0]+20, seating_top+76), (seating_rect[2]-20, seating_top+76)], fill="#FFFFFF", width=1)

# Seating chart card backgrounds for the three charts (rounded placeholders)
# Positions match detected icon crops so the icons will be pasted over them.
seat1 = (48, 2049, 48 + 462, 2049 + 437)
seat2 = (546, 2049, 546 + 462, 2049 + 437)
seat3 = (1044, 2049, 1044 + 396, 2049 + 437)
draw.rounded_rectangle(seat1, radius=16, fill="#FBFBFB")
draw.rounded_rectangle(seat2, radius=16, fill="#FBFBFB")
draw.rounded_rectangle(seat3, radius=16, fill="#FBFBFB")
# very subtle inner shadow/top highlight for each seating card
for s in (seat1, seat2, seat3):
    draw.line([(s[0]+8, s[1]+12), (s[2]-8, s[1]+12)], fill="#FFFFFF", width=1)
    draw.line([(s[0]+8, s[3]-12), (s[2]-8, s[3]-12)], fill="#EFEFEF", width=1)

# Thin separator below seating charts
seating_sep_y = seating_bottom
draw.line([(24, seating_sep_y), (1440 - 24, seating_sep_y)], fill="#E6E6E6", width=1)

# All events list area (background block and separators for list items)
events_top = seating_sep_y + 24
events_rect = (0, events_top, 1440, 2960)
draw.rectangle(events_rect, fill="#FFFFFF")
# Draw a few horizontal separators to represent list item divisions (icons/text will overlay)
list_y = events_top + 120
for i in range(5):
    draw.line([(24, list_y), (1440 - 24, list_y)], fill="#E9E9E9", width=1)
    list_y += 160

# Final bottom padding / subtle footer divider
draw.line([(24, 2960 - 160), (1440 - 24, 2960 - 160)], fill="#F0F0F0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/00_icon_Andrew_Schulz.png
try:
    _c0 = get_crop(0, 462, 437)
    canvas.paste(_c0, (546, 2049), _c0)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/01_icon_Seating_charts.png
try:
    _c1 = get_crop(1, 462, 437)
    canvas.paste(_c1, (48, 2049), _c1)
except Exception:
    pass
layout["Seating_charts"] = [48, 2049, 510, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/02_icon_Megan_Thee_Stallic.png
try:
    _c2 = get_crop(2, 396, 437)
    canvas.paste(_c2, (1044, 2049), _c2)
except Exception:
    pass
layout["Megan_Thee_Stallic"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/03_icon_S457.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (48, 1273), _c3)
except Exception:
    pass
layout["S457+"] = [48, 1273, 510, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/04_icon_7.45_my.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 84), _c4)
except Exception:
    pass
layout["7.45_my"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/05_icon_S382.png
try:
    _c5 = get_crop(5, 396, 533)
    canvas.paste(_c5, (1044, 1273), _c5)
except Exception:
    pass
layout["S382+"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 42, 58)
    canvas.paste(_c6, (1327, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1327, 4, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/07_icon_495.png
try:
    _c7 = get_crop(7, 204, 174)
    canvas.paste(_c7, (1236, 806), _c7)
except Exception:
    pass
layout["495"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/08_icon_S333.png
try:
    _c8 = get_crop(8, 462, 533)
    canvas.paste(_c8, (546, 1273), _c8)
except Exception:
    pass
layout["S333+"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/09_icon_Andrew_Schulz.png
try:
    _c9 = get_crop(9, 462, 437)
    canvas.paste(_c9, (546, 2049), _c9)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/10_icon_Capitals_at_Rangers.png
try:
    _c10 = get_crop(10, 462, 533)
    canvas.paste(_c10, (546, 1273), _c10)
except Exception:
    pass
layout["Capitals_at_Rangers_"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/11_icon_S333.png
try:
    _c11 = get_crop(11, 462, 533)
    canvas.paste(_c11, (546, 1273), _c11)
except Exception:
    pass
layout["S333+"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/12_icon_Megan_Thee_Stallic.png
try:
    _c12 = get_crop(12, 396, 437)
    canvas.paste(_c12, (1044, 2049), _c12)
except Exception:
    pass
layout["Megan_Thee_Stallic"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 39, 79)
    canvas.paste(_c13, (1401, 1304), _c13)
except Exception:
    pass
layout["icon_13"] = [1401, 1304, 1440, 1383]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/14_text_7.45_my.png
try:
    _c14 = get_crop(14, 151, 43)
    canvas.paste(_c14, (20, 15), _c14)
except Exception:
    pass
layout["7.45_my"] = [20, 15, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/15_text_St.png
try:
    _c15 = get_crop(15, 29, 27)
    canvas.paste(_c15, (800, 106), _c15)
except Exception:
    pass
layout["St"] = [800, 106, 829, 133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/16_text_St.png
try:
    _c16 = get_crop(16, 27, 27)
    canvas.paste(_c16, (1221, 125), _c16)
except Exception:
    pass
layout["St"] = [1221, 125, 1248, 152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/17_text_495.png
try:
    _c17 = get_crop(17, 36, 19)
    canvas.paste(_c17, (415, 167), _c17)
except Exception:
    pass
layout["495"] = [415, 167, 451, 186]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/18_text_St.png
try:
    _c18 = get_crop(18, 30, 27)
    canvas.paste(_c18, (966, 169), _c18)
except Exception:
    pass
layout["St"] = [966, 169, 996, 196]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/19_text_495.png
try:
    _c19 = get_crop(19, 43, 27)
    canvas.paste(_c19, (578, 254), _c19)
except Exception:
    pass
layout["495"] = [578, 254, 621, 281]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/20_text_St.png
try:
    _c20 = get_crop(20, 30, 27)
    canvas.paste(_c20, (712, 273), _c20)
except Exception:
    pass
layout["St"] = [712, 273, 742, 300]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/21_text_St.png
try:
    _c21 = get_crop(21, 27, 27)
    canvas.paste(_c21, (1089, 266), _c21)
except Exception:
    pass
layout["St"] = [1089, 266, 1116, 293]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/22_text_08.png
try:
    _c22 = get_crop(22, 20, 21)
    canvas.paste(_c22, (1331, 560), _c22)
except Exception:
    pass
layout["08"] = [1331, 560, 1351, 581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/23_text_495.png
try:
    _c23 = get_crop(23, 39, 19)
    canvas.paste(_c23, (1356, 597), _c23)
except Exception:
    pass
layout["495"] = [1356, 597, 1395, 616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/24_text_St.png
try:
    _c24 = get_crop(24, 29, 29)
    canvas.paste(_c24, (608, 608), _c24)
except Exception:
    pass
layout["St"] = [608, 608, 637, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/25_text_Ist_St.png
try:
    _c25 = get_crop(25, 64, 32)
    canvas.paste(_c25, (129, 622), _c25)
except Exception:
    pass
layout["Ist_St"] = [129, 622, 193, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/26_text_68.png
try:
    _c26 = get_crop(26, 26, 18)
    canvas.paste(_c26, (190, 676), _c26)
except Exception:
    pass
layout["68"] = [190, 676, 216, 694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/27_text_St.png
try:
    _c27 = get_crop(27, 30, 30)
    canvas.paste(_c27, (860, 749), _c27)
except Exception:
    pass
layout["St"] = [860, 749, 890, 779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/28_text_Madison_Square_Garden.png
try:
    _c28 = get_crop(28, 72, 72)
    canvas.paste(_c28, (408, 1297), _c28)
except Exception:
    pass
layout["Madison_Square_Garden"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/29_text_New_York_NY.png
try:
    _c29 = get_crop(29, 304, 57)
    canvas.paste(_c29, (42, 942), _c29)
except Exception:
    pass
layout["New_York,_NY"] = [42, 942, 346, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/30_text_Popular_events.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (408, 1297), _c30)
except Exception:
    pass
layout["Popular_events"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/31_text_E_Conf_Ist_Rnd_76e.png
try:
    _c31 = get_crop(31, 396, 533)
    canvas.paste(_c31, (1044, 1273), _c31)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76e"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/32_text_at_Knicks_Gm_5_H.png
try:
    _c32 = get_crop(32, 396, 533)
    canvas.paste(_c32, (1044, 1273), _c32)
except Exception:
    pass
layout["at_Knicks_(Gm_5,H"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/33_text_Tue.png
try:
    _c33 = get_crop(33, 95, 50)
    canvas.paste(_c33, (1037, 1756), _c33)
except Exception:
    pass
layout["Tue,"] = [1037, 1756, 1132, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/34_text_30_Time_T.png
try:
    _c34 = get_crop(34, 216, 48)
    canvas.paste(_c34, (1210, 1755), _c34)
except Exception:
    pass
layout["30,_Time_T"] = [1210, 1755, 1426, 1803]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/35_text_Seating_charts.png
try:
    _c35 = get_crop(35, 390, 76)
    canvas.paste(_c35, (39, 1920), _c35)
except Exception:
    pass
layout["Seating_charts"] = [39, 1920, 429, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/36_text_Billy_Joel.png
try:
    _c36 = get_crop(36, 191, 62)
    canvas.paste(_c36, (38, 2419), _c36)
except Exception:
    pass
layout["Billy_Joel"] = [38, 2419, 229, 2481]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/37_text_All_events.png
try:
    _c37 = get_crop(37, 256, 57)
    canvas.paste(_c37, (46, 2604), _c37)
except Exception:
    pass
layout["All_events"] = [46, 2604, 302, 2661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/38_text_Tonight.png
try:
    _c38 = get_crop(38, 172, 57)
    canvas.paste(_c38, (42, 2747), _c38)
except Exception:
    pass
layout["Tonight"] = [42, 2747, 214, 2804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/39_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_HG_2.png
try:
    _c39 = get_crop(39, 1440, 241)
    canvas.paste(_c39, (0, 2687), _c39)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/40_text_Mon_7.30_PM.png
try:
    _c40 = get_crop(40, 261, 53)
    canvas.paste(_c40, (45, 2819), _c40)
except Exception:
    pass
layout["Mon_7.30_PM"] = [45, 2819, 306, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/41_text_S363.png
try:
    _c41 = get_crop(41, 116, 52)
    canvas.paste(_c41, (345, 2817), _c41)
except Exception:
    pass
layout["S363"] = [345, 2817, 461, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/42_text_Madison_Square_Garden.png
try:
    _c42 = get_crop(42, 1440, 241)
    canvas.paste(_c42, (0, 2687), _c42)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/43_text_New_York.png
try:
    _c43 = get_crop(43, 197, 50)
    canvas.paste(_c43, (1011, 2822), _c43)
except Exception:
    pass
layout["New_York,"] = [1011, 2822, 1208, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/44_text_NY.png
try:
    _c44 = get_crop(44, 59, 40)
    canvas.paste(_c44, (1218, 2825), _c44)
except Exception:
    pass
layout["NY"] = [1218, 2825, 1277, 2865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/45_clickable_Displays_the_location_of_Madison_Square_.png
try:
    _c45 = get_crop(45, 1440, 704)
    canvas.paste(_c45, (0, 72), _c45)
except Exception:
    pass
layout["Displays_the_location_of_"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/46_clickable_Tracking.png
try:
    _c46 = get_crop(46, 144, 144)
    canvas.paste(_c46, (1260, 84), _c46)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_05_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-8/47_clickable_Tracking.png
try:
    _c47 = get_crop(47, 72, 72)
    canvas.paste(_c47, (906, 1297), _c47)
except Exception:
    pass
layout["Tracking"] = [906, 1297, 978, 1369]
