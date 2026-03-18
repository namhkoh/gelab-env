# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_03
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6.png
# step_index: 3/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 80)], fill="#ececec")

# Subtle toolbar area under status bar (background continuation)
draw.rectangle([(0, 80), (1440, 120)], fill="#f7f7f7")

# Search bar background (rounded)
search_left = 48
search_top = 120
search_right = 1440 - 48
search_bottom = search_top + 144  # 264
search_radius = 20
try:
    draw.rounded_rectangle(
        [(search_left, search_top), (search_right, search_bottom)],
        radius=search_radius,
        fill="#fafafa",
        outline=None
    )
except AttributeError:
    # Fallback: draw a rectangular background if rounded_rectangle unavailable
    draw.rectangle([(search_left, search_top), (search_right, search_bottom)], fill="#fafafa")

# Soft shadow / subtle bottom edge under search bar
draw.line([(search_left + 6, search_bottom + 8), (search_right - 6, search_bottom + 8)], fill="#ededed", width=1)
draw.line([(0, search_bottom + 48), (1440, search_bottom + 48)], fill="#e9e9e9", width=1)

# Divider under the search area (thin)
divider_y = search_bottom + 48  # ~312
draw.line([(48, divider_y), (1440 - 48, divider_y)], fill="#e6e6e6", width=1)

# Card-like subtle background for the "Recent searches" list
recent_top = 471
recent_item_height = 168
recent_count = 5
recent_bottom = recent_top + recent_item_height * recent_count  # end of list
card_left = 24
card_right = 1440 - 24
card_radius = 12
# Very subtle card outline to separate from page (no fill to avoid duplicating content)
try:
    draw.rounded_rectangle(
        [(card_left, recent_top - 10), (card_right, recent_bottom + 10)],
        radius=card_radius,
        fill=None,
        outline="#f5f5f5",
        width=1
    )
except AttributeError:
    draw.rectangle([(card_left, recent_top - 10), (card_right, recent_bottom + 10)], outline="#f5f5f5", width=1)

# Separator line after recent searches section
sep_y = recent_bottom + 10 + 30  # around 1311-ish
draw.line([(32, sep_y), (1440 - 32, sep_y)], fill="#e6e6e6", width=2)

# Suggestions section subtle grouping (rounded rect outline)
suggestions_top = 1423
suggestions_bottom = suggestions_top + 600  # leave space for items
try:
    draw.rounded_rectangle(
        [(40, suggestions_top - 20), (1400, suggestions_top + 300)],
        radius=14,
        fill=None,
        outline="#fbfbfb",
        width=1
    )
except AttributeError:
    draw.rectangle([(40, suggestions_top - 20), (1400, suggestions_top + 300)], outline="#fbfbfb", width=1)

# Bottom navigation background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill="#e6e6e6", width=1)

# Gentle page background tint (very light, to match app subtle off-white)
# only apply as a full-bleed behind content to match screenshot tone
draw.rectangle([(0, 0), (1440, 2960)], outline=None, fill=None)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/00_icon_Boston_Celtics.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/01_icon_The_Lion_King.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["The_Lion_King"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/02_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 639), _c2)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/03_icon_Wicked.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 975), _c3)
except Exception:
    pass
layout["Wicked"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/04_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 471), _c4)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 45, 70)
    canvas.paste(_c5, (1154, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/06_icon_Tracking.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/08_icon_6.49_W.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["6.49_W"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 65, 61)
    canvas.paste(_c9, (243, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 3, 308, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/11_icon_The_Phantom_of_the_Opera.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/12_icon_Just_Announced_by_My_Performers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1688), _c12)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 95, 69)
    canvas.paste(_c13, (1217, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/15_icon_The_Phantom_of_the_Opera.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 975), _c15)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 59)
    canvas.paste(_c16, (315, 6), _c16)
except Exception:
    pass
layout["icon_16"] = [315, 6, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/17_icon_Events_by_My_Performers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1520), _c17)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/18_icon_Wicked.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1143), _c18)
except Exception:
    pass
layout["Wicked"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/19_icon_Recent_searches.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 471), _c19)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 45, 66)
    canvas.paste(_c21, (1327, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/22_icon_6.49_W.png
try:
    _c22 = get_crop(22, 46, 62)
    canvas.paste(_c22, (187, 2), _c22)
except Exception:
    pass
layout["6.49_W"] = [187, 2, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/25_icon_Boston_Celtics.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1143), _c25)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/26_icon_Search.png
try:
    _c26 = get_crop(26, 288, 162)
    canvas.paste(_c26, (288, 2792), _c26)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/27_text_6.49_W.png
try:
    _c27 = get_crop(27, 149, 45)
    canvas.paste(_c27, (22, 13), _c27)
except Exception:
    pass
layout["6.49_W"] = [22, 13, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_03_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-6/30_text_Just_Announced_by_My_Performers.png
try:
    _c30 = get_crop(30, 1440, 168)
    canvas.paste(_c30, (0, 1856), _c30)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]
