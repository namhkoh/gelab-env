# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_03
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5.png
# step_index: 3/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for Eventbrite "Pets" search screen
# Assumes variables provided: canvas (1440x2960 PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Ensure full-white base
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# ---------- Status bar ----------
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(196, 196, 196))  # light gray status bar

# subtle darker top line for status bar separation
draw.line([(0, STATUS_H - 1), (1440, STATUS_H - 1)], fill=(180, 180, 180), width=1)

# ---------- Header / Search area ----------
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 168
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill=(255, 255, 255))

# Blue underline under search field (active indicator)
UNDERLINE_Y = HEADER_BOTTOM
draw.rectangle([(48, UNDERLINE_Y - 2), (1392, UNDERLINE_Y + 2)], fill=(43, 107, 230))  # vivid blue thin underline

# subtle divider below underline
draw.line([(0, UNDERLINE_Y + 4), (1440, UNDERLINE_Y + 4)], fill=(240, 240, 240), width=1)

# ---------- "Popular" section area background ----------
# Keep overall page white but add faint left accent bar for the Popular section header area
POPULAR_Y = 240
draw.rectangle([(0, POPULAR_Y - 40), (1440, POPULAR_Y + 6)], fill=(255, 255, 255))
# subtle divider above the Popular section
draw.line([(48, POPULAR_Y - 18), (1392, POPULAR_Y - 18)], fill=(240, 240, 240), width=1)

# Draw separators for the list items under "Popular"
popular_item_ys = [340, 460, 580, 700, 820]  # approximate y positions for popular quick-links
for y in popular_item_ys:
    draw.line([(48, y + 58), (1392, y + 58)], fill=(245, 245, 246), width=1)

# ---------- Events header separator ----------
EVENTS_HEADER_Y = 1026
draw.line([(48, EVENTS_HEADER_Y), (1392, EVENTS_HEADER_Y)], fill=(235, 235, 238), width=1)

# ---------- Event cards (rounded) ----------
# Use detected event card x & width: left margin 48, width 1344 -> right = 1392
CARD_LEFT = 48
CARD_RIGHT = 1392
CARD_WIDTH = CARD_RIGHT - CARD_LEFT
CARD_RADIUS = 12

# Event card top positions (derived from detected elements)
card_tops = [1117, 1513, 1909, 2305]  # y positions from detected elements
card_height = 360

# A faint drop shadow and white card for each event entry
for y in card_tops:
    # shadow
    shadow_rect = [(CARD_LEFT + 6, y + 10), (CARD_RIGHT + 6, y + card_height + 10)]
    draw.rounded_rectangle(shadow_rect, radius=CARD_RADIUS, fill=(240, 240, 242))
    # main card
    card_rect = [(CARD_LEFT, y), (CARD_RIGHT, y + card_height)]
    draw.rounded_rectangle(card_rect, radius=CARD_RADIUS, fill=(255, 255, 255), outline=None)

    # top divider for card (very light)
    draw.line([(CARD_LEFT + 12, y + 12), (CARD_RIGHT - 12, y + 12)], fill=(248, 248, 249), width=1)

    # left thumbnail background (subtle pale placeholder behind images)
    thumb_x0 = CARD_LEFT + 12
    thumb_y0 = y + 18
    thumb_x1 = thumb_x0 + 240
    thumb_y1 = thumb_y0 + 240
    draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], radius=10, fill=(242, 240, 244))

    # faint vertical separator between thumbnail and text area
    draw.line([(thumb_x1 + 14, thumb_y0), (thumb_x1 + 14, thumb_y1)], fill=(245, 245, 246), width=1)

    # subtle bottom divider for the card
    draw.line([(CARD_LEFT + 8, y + card_height + 2), (CARD_RIGHT - 8, y + card_height + 2)], fill=(243, 243, 245), width=1)

# ---------- Small gray separators between event cards (extra) ----------
for y in [card_tops[0] + card_height + 18, card_tops[1] + card_height + 18, card_tops[2] + card_height + 18]:
    draw.line([(CARD_LEFT + 8, y), (CARD_RIGHT - 8, y)], fill=(245, 245, 246), width=1)

# ---------- Floating colored banner for first few events (background only) ----------
# Add a subtle warm accent behind event date areas (do not draw any text)
for idx, y in enumerate(card_tops):
    # small orange accent rectangle near top-left of card text area
    accent_x0 = CARD_LEFT + 280
    accent_y0 = y + 22
    accent_x1 = accent_x0 + 220
    accent_y1 = accent_y0 + 28
    draw.rectangle([(accent_x0, accent_y0), (accent_x1, accent_y1)], fill=(230, 92, 20))

# ---------- Bottom navigation bar ----------
NAV_TOP = 2804
NAV_BOTTOM = 2960
draw.rectangle([(0, NAV_TOP), (1440, NAV_BOTTOM)], fill=(255, 255, 255))
# top border
draw.line([(0, NAV_TOP), (1440, NAV_TOP)], fill=(230, 230, 232), width=1)

# active item background circle (behind the second icon segment) - background only
segment_w = 1440 / 5
active_seg = 1  # 0-based index, second segment is active in screenshot
cx = int(segment_w * active_seg + segment_w / 2)
cy = int((NAV_TOP + NAV_BOTTOM) / 2)
r = 38
draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(249, 123, 31))  # orange highlight behind active icon

# subtle nav icon backdrop for center heart (just background shapes, icons will be pasted)
center_seg = 2
cx2 = int(segment_w * center_seg + segment_w / 2)
r2 = 26
draw.ellipse([(cx2 - r2, cy - r2), (cx2 + r2, cy + r2)], fill=(255, 255, 255))  # white backdrop

# final top shadow for nav
draw.line([(0, NAV_TOP + 1), (1440, NAV_TOP + 1)], fill=(245, 245, 246), width=1)

# ---------- Page left gutter accent (subtle) ----------
# very light vertical guide to echo card margins (purely decorative)
draw.rectangle([(24, UNDERLINE_Y + 6), (28, NAV_TOP - 6)], fill=(250, 250, 251))

# Done - UI background, headers, cards, separators and nav backgrounds rendered.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/00_icon_City_..png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["City_."] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/01_icon_Petsl.png
try:
    _c1 = get_crop(1, 173, 111)
    canvas.paste(_c1, (187, 112), _c1)
except Exception:
    pass
layout["Petsl"] = [187, 112, 360, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/02_icon_Pet_Portrait_1_00-4_00_PM.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1909), _c2)
except Exception:
    pass
layout["Pet_Portrait_1:00-4:00_PM"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/03_icon_4.47.png
try:
    _c3 = get_crop(3, 56, 59)
    canvas.paste(_c3, (115, 4), _c3)
except Exception:
    pass
layout["4.47"] = [115, 4, 171, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/04_icon_4.47.png
try:
    _c4 = get_crop(4, 53, 59)
    canvas.paste(_c4, (184, 3), _c4)
except Exception:
    pass
layout["4.47"] = [184, 3, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/05_icon_Alexandra_Schmeling.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1117), _c5)
except Exception:
    pass
layout["Alexandra_Schmeling"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 55, 58)
    canvas.paste(_c6, (313, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [313, 4, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 42, 53)
    canvas.paste(_c7, (254, 7), _c7)
except Exception:
    pass
layout["icon_7"] = [254, 7, 296, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/08_icon_4.47.png
try:
    _c8 = get_crop(8, 108, 108)
    canvas.paste(_c8, (62, 116), _c8)
except Exception:
    pass
layout["4.47"] = [62, 116, 170, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/09_icon_Fr.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Fr"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/10_icon_Paint_Your_Pet_Custom_Pet_Portraits_wl.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1513), _c10)
except Exception:
    pass
layout["Paint_Your_Pet:_Custom_Pe"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/11_icon_semo_pets.png
try:
    _c11 = get_crop(11, 1344, 120)
    canvas.paste(_c11, (48, 738), _c11)
except Exception:
    pass
layout["semo_pets"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/12_icon_4.47.png
try:
    _c12 = get_crop(12, 89, 57)
    canvas.paste(_c12, (17, 5), _c12)
except Exception:
    pass
layout["4.47"] = [17, 5, 106, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 47, 61)
    canvas.paste(_c13, (1322, 2), _c13)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 92, 63)
    canvas.paste(_c14, (1216, 0), _c14)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1308, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/15_icon_8_11_creator_followers.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["8_11_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1099, 96), _c16)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 149, 144)
    canvas.paste(_c17, (1243, 97), _c17)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/18_icon_WIIH.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1513), _c18)
except Exception:
    pass
layout["WIIH"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/19_icon_Pet_Portrait_1_00-4_00_PM.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1909), _c19)
except Exception:
    pass
layout["Pet_Portrait_1:00-4:00_PM"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/20_icon_More.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (1152, 2804), _c20)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/22_icon_I_O0_PM_EST.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1909), _c22)
except Exception:
    pass
layout["I:O0_PM_EST"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/23_icon_Alexandra_Schmeling.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1513), _c23)
except Exception:
    pass
layout["Alexandra_Schmeling"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/24_icon_semo_pets.png
try:
    _c24 = get_crop(24, 90, 94)
    canvas.paste(_c24, (35, 767), _c24)
except Exception:
    pass
layout["semo_pets"] = [35, 767, 125, 861]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/25_icon_Paint_Your_Pet_Custom_Pet_Portraits_wl.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1117), _c25)
except Exception:
    pass
layout["Paint_Your_Pet:_Custom_Pe"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/26_icon_Day.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 2305), _c26)
except Exception:
    pass
layout["Day"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/27_icon_Events.png
try:
    _c27 = get_crop(27, 84, 85)
    canvas.paste(_c27, (38, 892), _c27)
except Exception:
    pass
layout["Events"] = [38, 892, 122, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/28_text_Popular.png
try:
    _c28 = get_crop(28, 221, 78)
    canvas.paste(_c28, (44, 298), _c28)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/29_text_wallis_annenberg_petspace.png
try:
    _c29 = get_crop(29, 1344, 120)
    canvas.paste(_c29, (48, 378), _c29)
except Exception:
    pass
layout["wallis_annenberg_petspace"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/30_text_austin_pets_alive.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 498), _c30)
except Exception:
    pass
layout["austin_pets_alive"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/31_text_bets_for_pets.png
try:
    _c31 = get_crop(31, 241, 53)
    canvas.paste(_c31, (164, 671), _c31)
except Exception:
    pass
layout["bets_for_pets"] = [164, 671, 405, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/32_text_halloween_pets.png
try:
    _c32 = get_crop(32, 290, 51)
    canvas.paste(_c32, (164, 911), _c32)
except Exception:
    pass
layout["halloween_pets"] = [164, 911, 454, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/33_text_Events.png
try:
    _c33 = get_crop(33, 191, 61)
    canvas.paste(_c33, (45, 1026), _c33)
except Exception:
    pass
layout["Events"] = [45, 1026, 236, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/34_text_Sat_May_4_._8_00_AM_EDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2305), _c34)
except Exception:
    pass
layout["Sat,_May_4_._8:00_AM_EDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/35_text_Parade.png
try:
    _c35 = get_crop(35, 163, 50)
    canvas.paste(_c35, (391, 2481), _c35)
except Exception:
    pass
layout["Parade"] = [391, 2481, 554, 2531]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/36_text_of_Seat_Pleasant.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 2305), _c36)
except Exception:
    pass
layout["of_Seat_Pleasant"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/37_text_8_11_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2305), _c37)
except Exception:
    pass
layout["8_11_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/38_text_Fr.png
try:
    _c38 = get_crop(38, 39, 12)
    canvas.paste(_c38, (961, 2794), _c38)
except Exception:
    pass
layout["Fr"] = [961, 2794, 1000, 2806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/39_clickable_Pets.png
try:
    _c39 = get_crop(39, 1344, 191)
    canvas.paste(_c39, (48, 72), _c39)
except Exception:
    pass
layout["Pets"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/40_clickable_bets_for_pets.png
try:
    _c40 = get_crop(40, 1344, 120)
    canvas.paste(_c40, (48, 618), _c40)
except Exception:
    pass
layout["bets_for_pets"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/41_clickable_halloween_pets.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 858), _c41)
except Exception:
    pass
layout["halloween_pets"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_03_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-5/42_clickable_Favorites.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
