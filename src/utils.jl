## Copied from Distributions.jl
## macro for argument checking
macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(
                ArgumentError(
                    string(
                        $(string(D)), ": the condition ", $(string(cond)),
                        " is not satisfied."),
                ),
            )
        end
    end
end

# rowlogsumexps
rowlogsumexp(x) = logsumexp(x; dims=2)

# weighted median
weighted_median(x, w) = Statistics.median(convert(Array{Float64}, x), 
    StatsBase.weights(convert(Array{Float64}, w)))

    
# count parameters for AIC and BIC
function _count_π(π_list)
    return length(π_list) - 1
end

function _count_P_mat(P_mat)
    return size(P_mat, 1) * (size(P_mat, 1) - 1)
end

function _count_Q_mat(Q_mat)
    return (size(Q_mat, 1) - 1) * (size(Q_mat, 1) - 1) 
end

function _count_params(state_list)
    return sum(length.(HMMToolkit.params.(state_list)))
end


# given a dataframe ts, with time_since and time_interval, 
# linear interpolate each column indicated in response_list
function linear_interpolate(ts, response_list; ID_list = ["ID"])  # checked

    # Calculate the slopes for each response column
    slopes = DataFrame()
    for col in response_list
        slope_col_name = Symbol(string(col, "_slope"))
        slopes[!, slope_col_name] = (ts[!, col] - ShiftedArrays.lag(ts[!, col])) ./ ts.time_interval
    end

    # Create a new DataFrame to hold interpolated values
    ts_new = DataFrame()

    for i in 1:nrow(ts)
        if (i > 1) && (ts.time_interval[i] > 1)
            # just append row 1; else is the time interval > 1? if yes then interpolate
            timex = 1 : ts.time_interval[i]
            new_rows = DataFrame()
            new_rows.time_since = timex .+ ts.time_since[i-1]
            
            for col in response_list
                slope_col_name = Symbol(string(col, "_slope"))
                new_rows[!, col] = slopes[!, slope_col_name][i] .* timex .+ ts[!, col][i-1]
            end

            ts_new = vcat(ts_new, new_rows)
        else
            ts_new = vcat(ts_new, DataFrame(ts[i, vcat(response_list, "time_since")]))
        end
    end

    for col in ID_list
        ts_new[!, col] .= ts[1, col]
    end

    return ts_new
end


# replace nan by a number
function nan2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), g, x[i])
    end
end

# replace inf by a number
function inf2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isinf(x[i]), g, x[i])
    end
end