#'
#' @param se1e2iiar_2strains_pop Data frame containing `location_code`, `S`, `E1`, `E2`, `I`, `Ia`, `R`, `E1_b`, `E2_b`, `I_b`, and`Ia_b` for the entire population
#' @param mobility_matrix List (1 entry for each time period) with each entry containing a data frame with `from`, `to`, `n` for the number of people who travel. Each data frame must be complete.
#' @param dynamic_seeds Data.table containing `location_code`, `day`, `n`, `n_b` for dynamic seeding (or `NULL`)
#' @param betas Data table with `locaton code`, `day`, `time`, `beta` (1 entry for each location for each time period). Float, infection parameter, 0.6
#' @param inputSeed Integer with seed for cpp code
#' @param latent_period Float, 3.0
#' @param presymptomatic_period Float 2.0
#' @param infectious_period Float, 5.0
#' @param b_relative_infectiousness Float, relative infectiousness of strain B
#' @param presymptomatic_relative_infectiousness Float, relative infectiousness of presymptomatic
#' @param asymptomatic_prob Float, Proportion/probability of asymptomatic given infectious
#' @param asymptomatic_relative_infectiousness Float, Relative infectiousness of asymptomatic infectious
#' @param N Int = 1 int, Number of internal simulations (average taken). This is generally used for parameter fitting.
#' @return A data.table containing the following variables:


def asymmetric_mobility_se1e2iiar( se1e2iiar_pop = spread::asymmetric_mobility_se1e2iiar_dummy_se1e2iiar_pop,
                                          mobility_matrix = spread::asymmetric_mobility_se1e2iiar_dummy_mobility_matrix,
                                          dynamic_seeds = NULL,
                                          betas = spread::asymmetric_mobility_se1e2iiar_dummy_betas,
                                          inputSeed = as.numeric(Sys.time()),
                                          latent_period = 3.0,
                                          presymptomatic_period = 2.0,
                                          infectious_period = 5.0,
                                          presymptomatic_relative_infectiousness = 1.25,
                                          asymptomatic_prob = 0.4,
                                          asymptomatic_relative_infectiousness = 0.5,
                                          N = 1) {
  stopifnot(length(mobility_matrix) >= length(unique(betas$day)))