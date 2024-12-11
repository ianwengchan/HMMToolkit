function compute_vsax_breakpoints(trip_list, response_list)  # checked

    ## INPUT
    # trip_list: list of one-dimensional time-series
    # response_list

    ## OUTPUT
    # vsax_cuts: vsax break-points for trip_list

    joined_T = reduce(vcat, trip_list)
    num_dim = length(response_list)
    b1 = fill(- Inf, num_dim)
    b2 = [quantile(skipmissing(joined_T[!, response_list[i]]), 0.05) for i in 1:num_dim]
    b3 = [quantile(skipmissing(joined_T[!, response_list[i]]), 0.15) for i in 1:num_dim]
    b4 = [quantile(skipmissing(joined_T[!, response_list[i]]), 0.85) for i in 1:num_dim]
    b5 = [quantile(skipmissing(joined_T[!, response_list[i]]), 0.95) for i in 1:num_dim]
    b6 = fill(+ Inf, num_dim)

    return vcat(b1', b2', b3', b4', b5', b6')

end


function get_vsax_symbol(paa_k, vsax_cuts)

    ## INPUT
    # paa_k: discrete value for window k
    # vsax_cuts: vsax break-points

    ## OUTPUT
    # αi: symbol such that paa_k in (bi, b{i+1})

    return ["A", "B", "C", "D", "E"][findfirst(paa_k .<= vsax_cuts) - 1]

end


function get_vsax_sequence_from_trip(ts, l_size, vsax_cuts, response_list)  # checked

    ## INPUTS
    # ts: a trip time-series, with best_state decoded from a CTHMM

    ## OUTPUTS
    # vsaxseq_ts
        ### attributes:
        # V: a vsax letter sequence representing the trip
        # length: a sequence of vsax letter lengths (for MDL calculation)
        # window: a sequence of windows, where tss_k corresponds to letter_k of V_ts
        # ID: ID of the trip

    num_dim = length(response_list)

    ID_ts = ts.ID[1]
    V_ts = Tuple[]
    length_ts = Int64[]
    window_ts = DataFrame[]  # Correct initialization of an empty array for DataFrames

    ts_size = size(ts, 1)  # nrow of ts
    c_previous = ""
    tss_previous = DataFrame()

    for k = 1:(ts_size - (l_size - 1))
        tss_k = ts[k:(k + l_size - 1), :]
        paa_k = mean.(eachcol(tss_k[!, response_list]))
        c_k = Tuple(get_vsax_symbol(paa_k[i], vsax_cuts[!, i]) for i in 1:num_dim)

        if isempty(V_ts) || (c_k != c_previous)
            letter_k = c_k
            push!(V_ts, letter_k)
            push!(window_ts, DataFrame(tss_k))
            push!(length_ts, 1)
        else
            # Concatenating data from previous row and current row into a new DataFrame
            tss_k = DataFrame(union(eachrow.([tss_previous, DataFrame(tss_k)])...))
            letter_k = c_k
            window_ts[end] = tss_k
            length_ts[end] = length_ts[end] + 1
        end

        c_previous = c_k
        tss_previous = window_ts[end]
    end

    return (V = V_ts, length = length_ts, window = window_ts, ID = ID_ts)
end


function get_vsax_sequence(trip_list, l_size, vsax_cuts, response_list)  # checked

    ## INPUTS
    # trip_list: a GroupedDataFrame of trip time-series

    ## OUTPUTS
    # vsaxseq_list
        ### attributes:
        # list: list of vsaxseq_ts (from `get_vsax_sequence_from_trip`)
        # V: vector of vsax letter sequences, one sequence per trip
        # length: vector of sequences of vsax letter lengths, one sequence per trip (for MDL calculation)
        # window: vector of window sequences, corresponding to V_list
        # ID: vector of trip IDs

    num_ts = length(trip_list)
    vsaxseq_list = Array{NamedTuple}(undef, num_ts)
    @threads for ts_idx in 1:num_ts
        ts = trip_list[ts_idx]
        vsaxseq_list[ts_idx] = get_vsax_sequence_from_trip(ts, l_size, vsax_cuts, response_list)
    end

    V_list = getfield.(vsaxseq_list, :V)
    length_list = getfield.(vsaxseq_list, :length)
    window_list = getfield.(vsaxseq_list, :window)
    ID_list = getfield.(vsaxseq_list, :ID)

    return (list = vsaxseq_list, V = V_list, length = length_list, window = window_list, ID = ID_list)

end


function get_vsax_words_w_from_trip(vsaxseq_ts, w)  # checked

    ## INPUTS:
    # vsaxseq_ts (from `get_vsax_sequence_from_trip`)
        ### attributes:
        # V: a vsax letter sequence representing the trip
        # length: a sequence of vsax letter lengths (for MDL calculation)
        # window: a sequence of windows, where tss_k corresponds to letter_k of V
        # ID: ID of the trip

    # w: size of vsax words

    ## OUTPUTS
    # vsaxwords_w_ts
        ### attributes:
        # words: a sequence of vsax words of size w for a trip
        # window: a sequence of windows, where tss_k corresponds to word_k
        # idx: a sequence of indices (location within a trip), where idx_k corresponds to word_k
        # ID: a sequence of trip ID, where ID_k corresponds to word_k
    
    words_w_ts = Vector{Tuple}[]
    # length_w_ts = []
    window_w_ts = DataFrame[]
    idx_w_ts = Int64[]
    ID_w_ts = Int64[]

    V_ts = vsaxseq_ts.V
    window_ts = vsaxseq_ts.window
    ID_ts = vsaxseq_ts.ID

    for k in 1:(length(V_ts) - (w - 1))

        # if (nrow(vcat(window_ts[k:k+w-1]...)) > 1)  # it is NOT a pattern if there is only 1 observation (row)
            
            push!(words_w_ts, V_ts[k:k+w-1])
            # push!(length_w_ts, length_ts[k:k+w-1])
            push!(window_w_ts, DataFrame(union(eachrow.(window_ts[k:k+w-1])...)))  # Concatenate DataFrames vertically
            push!(idx_w_ts, k)
            push!(ID_w_ts, ID_ts)

        # end
        
    end

    return (words = words_w_ts, window = window_w_ts, idx = idx_w_ts, ID = ID_w_ts)

end


function get_vsax_words_w(vsaxseq_list, w)  # checked

    ## INPUTS
    # vsaxseq_list (from `get_vsax_sequence`)
        ### attributes:
        # list: list of vsaxseq_ts (from `get_vsax_sequence_from_trip`)
        # V: vector of vsax letter sequences, one sequence per trip
        # length: vector of sequences of vsax letter lengths, one sequence per trip (for MDL calculation)
        # window: vector of window sequences, corresponding to V_list
        # ID: vector of trip IDs

    # w: size of vsax words

    ## OUTPUTS
    # vsaxwords_w
        ### attributes:
        # list: list of vsaxwords_w_ts (from `get_vsax_words_w_from_trip`)
        # words: a sequence of vsax words of size w from all trips
        # window: a sequence of windows, where tss_k corresponds to word_k
        # idx: a sequence of indices (location within a trip), where idx_k corresponds to word_k
        # ID: a sequence of trip ID, where ID_k corresponds to word_k

    num_ts = length(vsaxseq_list)
    words_w_list = Array{NamedTuple}(undef, num_ts)

    @threads for ts_idx in 1:num_ts

        vsaxseq_ts = vsaxseq_list[ts_idx]
        words_w_list[ts_idx] = get_vsax_words_w_from_trip(vsaxseq_ts, w)

    end

    # transform vector of vectors to a single vector
    words_w = reduce(vcat, getfield.(words_w_list, :words))
    window_w = reduce(vcat, getfield.(words_w_list, :window))
    idx_w = reduce(vcat, getfield.(words_w_list, :idx))
    ID_w = reduce(vcat, getfield.(words_w_list, :ID))

    return (list = words_w_list, words = words_w, window = window_w, idx = idx_w, ID = ID_w)

end


function plot_windows(windows, response_list; color_list = nothing, title = nothing, display_plot = true)  # checked

    ## INPUTS
    # windows: vector of windows to be plotted
    # response_list: list of columns in windows (DataFrames) to be plotted
    # color_list: (optional) list of linecolors for the plots
    # title: (optional) title for the plot

    ## DISPLAY (by default) and/or OUTPUT the plot p

    if isnothing(color_list)
        color_list = ["indianred1", "royalblue1", "grey", "lightgreen"]
    end

    p = plot(legend = false)

    for k in 1:length(windows)

        window_k = windows[k]

        timex = Base.copy(window_k.time_since)
        timex = timex .- timex[1]

        for q in 1:length(response_list)
            p = plot!(timex, window_k[!, response_list[q]], linecolor = color_list[q])
        end
    
    end

    if !isnothing(title)
        p = title!(title)
    end

    if (display_plot)
        display(p)
    end

    return p

end


function TripMD_filter_windows_by_duration(windows, words_idx, IDs; bounds = nothing)  # checked

    if isnothing(bounds)
        bounds = [20 80]
    end

    duration_list = nrow.(windows)

    duration_lb, duration_ub = percentile(duration_list, [20 80])
    idx = findall(x -> x >= duration_lb && x <= duration_ub, duration_list)

    return idx, windows[idx], words_idx[idx], IDs[idx]

end


function TripMD_filter_windows(windows, words_idx, IDs; 
    by_duration = true, duration_bounds = nothing)  # checked

    ## INPUTS
    # windows: vector of windows with a particular pattern
    # words_idx: vector of indices (location within trip), where idx_k corresponds to window_k
    # IDs: vector of trip IDs, where ID_k corresponds to window_k
    # ll_list, duration_list, count_list: outputs from `CTHMM_batch_decode_for_subjects_list_instance`

    ## OUTPUTS - same as INPUTS
    
    if (by_duration) && (size(windows, 1) > 1)  # filter by duration

        idx_filter, windows, words_idx, IDs = TripMD_filter_windows_by_duration(windows, words_idx, IDs; bounds = duration_bounds)

    end

    return windows, words_idx, IDs

end


function compute_dtw_distance_matrix(windows, response_list)  # checked

    ## INPUTS
    # windows: vector of windows (DataFrames) with a particular pattern;
    ### must include columns in reponse_list and ID_list (default ['ID']), time_interval, time_since
    # response_list: list of columns in windows (DataFrames) to be interpolated

    ## OUTPUTS
    # d_mat: dtw distance matrix of size num_windows-by-num_windows
    #### windows_itp: (deprecated) vector of corresponding windows, interpolated by default #####

    num_windows = size(windows, 1)

    ## create a distance matrix
    d_mat = zeros(num_windows, num_windows)

    for i in 1:num_windows

        for j in (i+1):num_windows

            d_mat[i, j] = DynamicAxisWarping.dtw(Matrix(windows[i][!, response_list])', 
            Matrix(windows[j][!, response_list])')[1]

            d_mat[j, i] = d_mat[i, j]

        end

    end

    return d_mat

end


function get_members(candidate_i, d_mat, windows, words_idx, IDs, w)

    ## INPUTS
    # candidate_i: index of the i-th candidate in the list of candidates
    # d_mat: dtw distance matrix of size num_windows-by-num_windows
    # words_idx: a sequence of indices (location within a trip), where idx_k corresponds to window_k
    # IDs: a sequence of trip ID, where ID_k corresponds to window_k
    # w: size of pattern (vsax words)

    ## OUTPUTS
    # d_mat: updated d_mat
    # pruned_members_idx: a sequence of member indices in the list of windows

    mask = trues(size(d_mat, 1))
    mask[candidate_i] = false
    not_indices = findall(mask)

    same_ID = findall(IDs[not_indices] .== IDs[candidate_i])

    if !isempty(same_ID)  # from the same trip, check overlapping

        candidate_window = windows[candidate_i]
        
        for k in same_ID

            intersection = DataFrame(intersect(eachrow.([candidate_window, windows[not_indices[k]]])...))

            if !isempty(intersection)  # overlapping, cannot be a member
                d_mat[candidate_i, not_indices[k]] = 0
            end

        end
    end

    unpruned_members_idx = findall(d_mat[candidate_i, :] .> 0)

    if length(unpruned_members_idx) > 1  # start pruning

        pruned_members_idx = [unpruned_members_idx[1]]

        for member_idx in unpruned_members_idx[2:end]

            last_member_idx = pruned_members_idx[end]
            last_member_ID = IDs[last_member_idx]
            last_member_window = collect(words_idx[last_member_idx] : words_idx[last_member_idx] + (w-1))
            last_member_dist = d_mat[candidate_i, last_member_idx]

            member_ID = IDs[member_idx]
            if member_ID == last_member_ID  # from the same trip, check overlapping

                member_window = collect(words_idx[member_idx] : words_idx[member_idx] + (w-1))
                if length(intersect(member_window, last_member_window)) > 0  # overlapping

                    member_dist = d_mat[candidate_i, member_idx]
                    if member_dist < last_member_dist  # new member wins

                        pruned_members_idx[end] = member_idx  # replace last member with new
                        d_mat[candidate_i, last_member_idx] = 0

                    end  # otherwise do nothing and keep last member
                else
                    push!(pruned_members_idx, member_idx)  # not overlapping, add new member
                end
            else
                push!(pruned_members_idx, member_idx)  # not overlapping, add new member
            end
        end
    else
        pruned_members_idx = unpruned_members_idx
    end

    return d_mat, pruned_members_idx

end


function get_motif(pattern, d_mat, windows, words_idx, IDs; filter = false, R_pct = nothing)  # checked

    ## INPUTS
    # pattern: the vsax word that all windows are following
    # d_mat: dtw distance matrix of size num_windows-by-num_windows
    # windows: vector of corresponding windows
    # words_idx: a sequence of indices (location within a trip), where idx_k corresponds to window_k
    # IDs: a sequence of trip ID, where ID_k corresponds to window_k

    ## OUTPUTS
    # motif
        ### attributes
        # m_pattern: the vsax word representing the motif
        # m_center: dataframe of motif center; can be nothing
        # m_members: dataframes of motif members; can be nothing
        # m_center_idx: index (location within a trip) of motif center; can be nothing
        # m_members_idx: indices (location within a trip) of motif members; can be nothing
        # m_center_ID: trip ID of motif center; can be nothing
        # m_members_ID: trip IDs of motif members; can be nothing
        # m_mean: average dtw distance between motif center and members; Inf if motif is empty

    if (filter == true)  # no further filtering by default
        if isnothing(R_pct)  # use median of d_mat values by default
            R_pct = 50
        end
        pct = percentile(vec(d_mat[d_mat .> 0]), R_pct)
        d_mat[d_mat .> pct] .= 0
    end

    m_center = nothing
    m_center_ID = nothing
    m_center_idx = nothing
    m_members = nothing
    m_members_ID = nothing
    m_members_idx = nothing
    m_mean = Inf

    # before checking for overlaps
    naive_count_members = vec(count(>(0), d_mat, dims = 2))
    naive_count_max = maximum(naive_count_members)

    w = length(pattern)

    if naive_count_max > 0
        # there is at least 1 candidate with at least 1 member (before checking for overlaps)

        candidates_idx = findall(naive_count_members .> 0)
        # candidates are all those with at least 1 member (before checking for overlaps)

        count_max = 0

        for candidate_i in candidates_idx

            if naive_count_members[candidate_i] >= count_max
                # naive_count >= count_new
                # => naive_count < count_max => count_new < count_max for sure

                # get members for candidate_i and update d_mat
                d_mat, pruned_members_idx = get_members(candidate_i, d_mat, windows, words_idx, IDs, w)
                count_new = length(pruned_members_idx)

                if count_new > count_max  # candidate_i has more members, update
                    count_max = count_new

                    m_center = windows[candidate_i]
                    m_members = windows[pruned_members_idx]
                    m_center_idx = words_idx[candidate_i]
                    m_members_idx = words_idx[pruned_members_idx]
                    m_center_ID = IDs[candidate_i]
                    m_members_ID = IDs[pruned_members_idx]
                    m_mean = sum(d_mat[candidate_i, :]) / count_new

                elseif count_new == count_max  # candidate_i has the same number of members
                    c_mean = sum(d_mat[candidate_i, :]) / count_new

                    if c_mean < m_mean  # candidate_i has a smaller mean distance, update

                        m_center = windows[candidate_i]
                        m_members = windows[pruned_members_idx]
                        m_center_idx = words_idx[candidate_i]
                        m_members_idx = words_idx[pruned_members_idx]
                        m_center_ID = IDs[candidate_i]
                        m_members_ID = IDs[pruned_members_idx]
                        m_mean = c_mean

                    end
                end
            end
        end
    end

    return (m_pattern = pattern, m_center = m_center, m_members = m_members, 
    m_center_idx = m_center_idx, m_members_idx = m_members_idx,
    m_center_ID = m_center_ID, m_members_ID = m_members_ID, m_mean = m_mean)

end


function TripMD_plot_motif(motif, response_list; 
    member_color = nothing, center_color = nothing, linewidth = 3, title = nothing, display_plot = true)  # checked

    if isnothing(title)
        if (typeof(motif.m_pattern) != String)
            string_rep = join([string(t) for t in motif.m_pattern], ", ")
        else
            string_rep = motif.m_pattern
        end
        title = "Motif: " * string_rep * ", count filtered"
    end

    if !isnothing(motif.m_center)  # motif is non-empty, do plot

        ## plot motif members
        p = plot_windows(motif.m_members, response_list; color_list = member_color, title = title, display_plot = false)

        ## plot motif center
        if isnothing(center_color)
            center_color = ["red", "blue", "black", "green"]
        end

        timex = Base.copy(motif.m_center.time_since)
        timex = timex .- timex[1]

        for q in 1:length(response_list)
            p = plot!(timex, motif.m_center[!, response_list[q]], linecolor = center_color[q], linewidth = linewidth)
        end

    end

    if (display_plot)
        display(p)
    end

    return p

end


function extract_motifs(trip_list, response_list, l_size, w_min; w_max = nothing)

    ## INPUTS
    # trip_list: a GroupedDataFrame of trip time-series
    ### must include best_state, columns in reponse_list and ID_list (default ['ID']), time_interval, time_since
    # response_list: list of columns in windows (DataFrames) to be considered
    # model: CTHMM model
    ### must include π_list, Q_mat and state_list
    # w_min: minimum size of vsax words
    # w_max: (optional) maximum size of vsax words; default to length of shortest trip
    
    ## OUTPUTS
    # motif_list: list of motifs (from `get_motif`)

    vsax_cuts = compute_vsax_breakpoints(trip_list, response_list)

    vsaxseq_list = get_vsax_sequence(trip_list, l_size, vsax_cuts, response_list)

    if (isnothing(w_max))
        w_max = minimum(length.(vsaxseq_list.V))
    end

    motif_list = NamedTuple[]

    w = w_min
    vsaxwords_w = get_vsax_words_w(vsaxseq_list.list, w)
    vsaxwords_w_unique = unique(vsaxwords_w.words)

    while ((length(vsaxwords_w) != length(vsaxwords_w_unique)) && (w <= w_max))
        # stop when there are no more repeating patterns (must)
        # or when w_max is reached

        for pattern in vsaxwords_w_unique

            idx = findall([this_word == pattern for this_word in vsaxwords_w.words])
            windows = vsaxwords_w.window[idx]
            words_idx = vsaxwords_w.idx[idx]
            IDs = vsaxwords_w.ID[idx]

            if length(windows) > 1  # if there is only 1 window, it is not a motif

                windows, words_idx, IDs = TripMD_filter_windows(windows, words_idx, IDs)

                if length(windows) > 1  # number of windows left after filtering

                    d_mat = compute_dtw_distance_matrix(windows, response_list)
                    motif = get_motif(pattern, d_mat, windows, words_idx, IDs)

                    if !isnothing(motif.m_center)  # motif is non-empty
                        push!(motif_list, motif)
                    end

                end
            end
        end

        w = w + 1
        vsaxwords_w = get_vsax_words_w(vsaxseq_list.list, w)
        vsaxwords_w_unique = unique(vsaxwords_w.words)
    end

    return motif_list
end




function find_segments(motif, vsaxseq_list)

    pointer = DataFrame(ID = vcat(motif.m_members_ID, motif.m_center_ID), 
                        idx = vcat(motif.m_members_idx, motif.m_center_idx))

    w = length(motif.m_pattern)

    num_ts = length(vsaxseq_list.V)
    segments_list = Array{Array{Array{Int64}}}(undef, num_ts)

    @threads for ts_idx = 1:num_ts
        ID_ts = vsaxseq_list.ID[ts_idx]
        pointer_ts = filter(:ID => ==(ID_ts), pointer)

        length_ts = vsaxseq_list.length[ts_idx]

        segments_ts = Array{Int64}[]
        split_idx = sort(unique(vcat(pointer_ts.idx, pointer_ts.idx .+ w)))
        previous_idx = 1

        if length(split_idx) > 0  # do split accordingly
            for index in split_idx
                push!(segments_ts, length_ts[previous_idx : (index-1)])
                previous_idx = index
            end
        end

        # Add the remaining part, or if there is only 1 segment (pattern DNE)
        if previous_idx <= length(length_ts)
            push!(segments_ts, length_ts[previous_idx : end])
        end

        segments_list[ts_idx] = segments_ts

    end

    return segments_list

end


function compute_mdl_cost(motif, vsaxseq_list)

    segments_list = find_segments(motif, vsaxseq_list)

    # length of segments
    ti_list = [sum.(segments_ts) for segments_ts in segments_list]

    # number of segments
    m_list = [length(segments_ts) for segments_ts in segments_list]

    # segmentation cost
    # DL3_list = [m_list[i] * log2(sum(ti_list[i])) for i in 1:length(segments_list)]
    DL3 = sum(m_list) * log2(sum(sum.(ti_list)))

    # data encoding cost
    DL1 = sum([sum(sum(-segments_list[ts_idx][i] .* log2.(segments_list[ts_idx][i] ./ ti_list[ts_idx][i])) 
    for i in 1:m_list[ts_idx]) for ts_idx in 1:length(segments_list)])    

    # parameter encoding cost
    DL2 = sum([sum(log2.(ti_list[i])) for i in 1:length(segments_list)])

    MDL_cost = DL1 + DL2 + DL3

    return MDL_cost

end


# function compute_mdl_cost(motif, V_list, ID_list)

#     # np = length of the pattern
#     # sp = number of unique vsax letters in the pattern
#     # DL_p = description length of the pattern

#     # na = length of sequence, with pattern replaced by a single letter
#     # sa = number of unique vsax letters in the sequence
#     # q = number of pattern appearance in the sequence
#     # DL = description length of the sequence given pattern

#     np = length(motif.m_pattern)
#     sp = length(unique(motif.m_pattern))
#     DL_p = log2(np) + np*log2(sp)

#     q_list = vcat([[i, count(==(i), motif.m_members_ID)]' for i in unique(ID_list)]...)
#     q_list = DataFrame(q_list, ["ID", "q"])

#     hat_C_list = V_list
#     @threads for i in 1:length(hat_C_list)
#         q = q_list.q[i]
#         if q > 0
#             hat_C_list[i] = replace(hat_C_list[i], motif.m_pattern => "0", count = q)
#         end
#     end

#     na = map(x -> length.(x), hat_C_list)
#     sa = map(x -> length.(unique.(x)), hat_C_list)

#     DL = log2.(na) .+ na .* log2.(sa .+ q)

#     MDL = DL .+ DL_p

#     return MDL

# end