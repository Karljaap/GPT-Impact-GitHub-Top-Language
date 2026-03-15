
//----------------------------------------------------------------------------//
//
// Proyecto: Tesis
// Impacto de ChatGPT en el número de programadores en GitHub
//
//----------------------------------------------------------------------------//

global path "c:/Users/ronco/Desktop/GPT-Impact-GitHub-Top-Language"

* Crear carpetas si no existen
cap mkdir "$path/output"
cap mkdir "$path/output/figures"
cap mkdir "$path/output/tables"

import delimited "$path/output/data/data_langs_balanced.csv", clear

sort unique_id year quarter
drop if iso2_code == "HK"
label var num_pushers_pc "Número de pushers por 100k habitantes"
label var gpt_available_post1 "ChatGPT Disponible"

**************************
// Cambio de nombres de etiquetas
**************************
replace language = "C_hashtag" if language == "C#"
replace language = "C_plus"    if language == "C++"

*****************
// Tesis_DataScience
*****************
local DataScience "C C_hashtag C_plus Go Java JavaScript PHP Python Ruby TypeScript"

foreach lang of local DataScience {

        local l`v' : variable label num_pushers_pc

        *-----------------------------------------------------
        * DID
        *-----------------------------------------------------
        eststo `lang'_did: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(did) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'did", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'did_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'did_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SC
        *-----------------------------------------------------
        eststo `lang'_sc: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sc) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sc", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sc_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sc_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SDID
        *-----------------------------------------------------
        eststo `lang'_sdid: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sdid) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sdid", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sdid_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sdid_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'
}

** Tabla con tres paneles

esttab C_did C_sc C_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                replace label booktabs                                                                   ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                delim("&")  ///
                nomtitle ///
                collabels(none) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                mgroups("\shortstack{DID}" ///
                                        "\shortstack{SC}" ///
                                        "\shortstack{SDID}",  ///
                                        pattern(1 1 1)                           ///
                                        prefix(\multicolumn{@span}{c}{) suffix(}) span                       ///
                                        erepeat(\cmidrule(lr){@span})) ///
                                nomtitles                       ///
                scalars("control_mean Media de referencia") ///
                        refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel A. \textbf{ \textit{C} } } }" , nolabel) ///
                        prefoot("") posthead(\hline) postfoot("")  nonumbers

esttab C_hashtag_did C_hashtag_sc C_hashtag_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel B. \textbf{ \textit{C\#} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab C_plus_did C_plus_sc C_plus_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel C. \textbf{ \textit{C++} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab Go_did Go_sc Go_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel D. \textbf{ \textit{Go} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab Java_did Java_sc Java_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel E. \textbf{ \textit{Java} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab JavaScript_did JavaScript_sc JavaScript_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel F. \textbf{ \textit{JavaScript} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab PHP_did PHP_sc PHP_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel G. \textbf{ \textit{PHP} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab Python_did Python_sc Python_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel H. \textbf{ \textit{Python} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab Ruby_did Ruby_sc Ruby_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel I. \textbf{ \textit{Ruby} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab TypeScript_did TypeScript_sc TypeScript_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel J. \textbf{ \textit{TypeScript} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

cd "$path"
